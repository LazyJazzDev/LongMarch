#include <gtest/gtest.h>

#include <cstdlib>
#include <new>

#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_sampler.h"
#include "grassland/rhi/rhi.h"

namespace {
thread_local bool fail_allocation = false;
thread_local size_t failure_size = 0;
thread_local bool failure_triggered = false;

class FailAllocation {
 public:
  explicit FailAllocation(size_t size = 0) {
    failure_size = size;
    failure_triggered = false;
    fail_allocation = true;
  }

  ~FailAllocation() {
    fail_allocation = false;
  }
};
}  // namespace

// Kept in a separate executable so fault injection cannot affect other RHI tests.
void *operator new(size_t size) {
  if (fail_allocation && (!failure_size || size == failure_size)) {
    fail_allocation = false;
    failure_triggered = true;
    throw std::bad_alloc{};
  }
  if (void *memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc{};
}

void operator delete(void *memory) noexcept {
  std::free(memory);
}

void operator delete(void *memory, size_t) noexcept {
  std::free(memory);
}

void *operator new[](size_t size) {
  return ::operator new(size);
}

void operator delete[](void *memory) noexcept {
  std::free(memory);
}

void operator delete[](void *memory, size_t) noexcept {
  std::free(memory);
}

namespace grassland::rhi {
using namespace backend::metal;

TEST(RhiMetalAllocation, FactoriesPropagateCpuAllocationFailures) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  bool propagated = false;
  {
    FailAllocation fail(sizeof(MetalAllocatedBuffer));
    try {
      ObjPtr<IBuffer> buffer = device->CreateBuffer({256});
    } catch (const std::bad_alloc &) {
      propagated = true;
    }
  }
  EXPECT_TRUE(failure_triggered);
  EXPECT_TRUE(propagated);
  propagated = false;
  {
    // Native sampler creation precedes the wrapper allocation; RAII must release
    // that state when wrapper allocation fails.
    FailAllocation fail(sizeof(MetalSampler));
    try {
      ObjPtr<ISampler> sampler = device->CreateSampler();
    } catch (const std::bad_alloc &) {
      propagated = true;
    }
  }
  EXPECT_TRUE(failure_triggered);
  EXPECT_TRUE(propagated);
  EXPECT_FALSE(device->CreateBuffer({0}));
  EXPECT_FALSE(device->CreateSampler({static_cast<SamplerFilter>(-1)}));
  ObjPtr<IBuffer> buffer = device->CreateBuffer({256});
  EXPECT_TRUE(buffer);
  ObjPtr<ISampler> sampler = device->CreateSampler();
  EXPECT_TRUE(sampler);
}

TEST(RhiMetalAllocation, FailedLeaseConstructionReturnsBorrowedUploadBuffer) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto &pool = static_cast<MetalDevice *>(device.Get())->UploadPool();
  auto warm = pool.Acquire(256);
  ASSERT_TRUE(warm);
  auto native = warm->NativeBuffer();
  warm.Reset();
  bool propagated = false;
  {
    FailAllocation fail(sizeof(MetalTemporalBufferPool::Lease));
    try {
      auto lease = pool.Acquire(256);
    } catch (const std::bad_alloc &) {
      propagated = true;
    }
  }
  EXPECT_TRUE(failure_triggered);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(pool.GetBufferCounts(256).second, 0u);
  auto lease = pool.Acquire(256);
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->NativeBuffer(), native);
  EXPECT_EQ(pool.GetBufferCounts(256).second, 1u);
}

TEST(RhiMetalAllocation, RecyclingAllocationFailureReleasesStorageWithoutThrowing) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto &pool = static_cast<MetalDevice *>(device.Get())->UploadPool();
  std::vector<backend::ObjectPtr<MetalTemporalBufferPool::Lease>> leases;
  // Enough decreasing occupancy samples to grow libc++'s deque storage.
  for (size_t i = 0; i < 1024; ++i) {
    leases.push_back(pool.Acquire(256));
    ASSERT_TRUE(leases.back());
  }
  bool exercised = false;
  for (auto &lease : leases) {
    const auto allocated = pool.GetBufferCounts(256).first;
    {
      FailAllocation fail;
      lease.Reset();
    }
    if (failure_triggered) {
      exercised = true;
      EXPECT_EQ(pool.GetBufferCounts(256).first, allocated - 1);
    }
  }
  EXPECT_TRUE(exercised);
  EXPECT_EQ(pool.GetBufferCounts(256).second, 0u);
  pool.Clear();
  EXPECT_EQ(pool.GetBufferCounts(256).first, 0u);
  EXPECT_TRUE(pool.Acquire(256));
}
}  // namespace grassland::rhi
