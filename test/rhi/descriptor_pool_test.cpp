#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <thread>

#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_transient_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {
TEST(D3D12DescriptorPool, ReleasedHolesAreReusedAndMergedOnBothSides) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12DescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8);
  auto a = pool.Acquire(2);
  auto b = pool.Acquire(2);
  auto c = pool.Acquire(2);
  auto d = pool.Acquire(2);
  ASSERT_TRUE(a && b && c && d);
  auto heap = a->Heap();
  auto start = a->CPU().ptr;
  auto hole = b->CPU().ptr;
  b.Reset();
  b = pool.Acquire(2);
  ASSERT_TRUE(b);
  EXPECT_EQ(b->Heap(), heap);
  EXPECT_EQ(b->CPU().ptr, hole);
  a.Reset();
  c.Reset();
  b.Reset();  // Merge both free neighbors while D remains alive.
  auto merged = pool.Acquire(6);
  ASSERT_TRUE(merged);
  EXPECT_EQ(merged->Heap(), heap);
  EXPECT_EQ(merged->CPU().ptr, start);
  d.Reset();
  merged.Reset();
  auto full = pool.Acquire(8);
  ASSERT_TRUE(full);
  EXPECT_EQ(full->CPU().ptr, start);
  full.Reset();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{1, 0}));
}

TEST(D3D12DescriptorPool, SmallestPagePrecedesTightestFit) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12DescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8);
  auto smaller = pool.Acquire(8);
  auto large = pool.Acquire(16);
  ASSERT_TRUE(smaller && large);
  auto small_heap = smaller->Heap();
  auto large_heap = large->Heap();
  smaller.Reset();
  large.Reset();
  large = pool.Acquire(12);
  ASSERT_TRUE(large);
  EXPECT_EQ(large->Heap(), large_heap);
  auto result = pool.Acquire(3);  // Prefer the 8-slot page over the larger page's 4-slot hole.
  ASSERT_TRUE(result);
  EXPECT_EQ(result->Heap(), small_heap);
}

TEST(D3D12DescriptorPool, TightestFitSplitsAndRegistersRemainder) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12DescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 16);
  auto a = pool.Acquire(6);
  auto separator = pool.Acquire(1);
  auto b = pool.Acquire(4);
  auto end = pool.Acquire(5);
  ASSERT_TRUE(a && separator && b && end);
  auto a_start = a->CPU().ptr;
  auto b_start = b->CPU().ptr;
  a.Reset();
  b.Reset();
  auto best = pool.Acquire(3);
  ASSERT_TRUE(best);
  EXPECT_EQ(best->CPU().ptr, b_start);
  auto remainder = pool.Acquire(1);
  ASSERT_TRUE(remainder);
  EXPECT_EQ(remainder->CPU().ptr,
            b_start + 3 * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV));
  auto other = pool.Acquire(5);
  ASSERT_TRUE(other);
  EXPECT_EQ(other->CPU().ptr, a_start);
  EXPECT_EQ(pool.GetPageCounts().first, 1u);
}

TEST(D3D12DescriptorPool, PagesGrowGeometricallyAndRemainAllocated) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12DescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 7);
  auto a = pool.Acquire(8);
  auto b = pool.Acquire(16);
  auto c = pool.Acquire(33);
  auto d = pool.Acquire(31);
  auto e = pool.Acquire(1);
  ASSERT_TRUE(a && b && c && d && e);
  EXPECT_EQ(a->Heap()->GetDesc().NumDescriptors, 8u);
  EXPECT_EQ(b->Heap()->GetDesc().NumDescriptors, 16u);
  EXPECT_EQ(c->Heap()->GetDesc().NumDescriptors, 64u);
  EXPECT_EQ(d->Heap(), c->Heap());
  EXPECT_EQ(e->Heap()->GetDesc().NumDescriptors, 128u);
  a.Reset();
  b.Reset();
  c.Reset();
  d.Reset();
  e.Reset();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{4, 0}));
  auto huge = pool.Acquire(8000);
  ASSERT_TRUE(huge);
  EXPECT_EQ(huge->Count(), 8000u);
  EXPECT_EQ(huge->Heap()->GetDesc().NumDescriptors, 8192u);
  EXPECT_EQ(huge->Heap()->GetDesc().Flags, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
  huge.Reset();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{5, 0}));
  EXPECT_FALSE(pool.Acquire(0));
  EXPECT_FALSE(pool.Acquire(UINT32_MAX));
  auto reused = pool.Acquire(8);
  ASSERT_TRUE(reused);
  EXPECT_EQ(reused->Heap()->GetDesc().NumDescriptors, 8u);
}

TEST(D3D12DescriptorPool, RandomizedAllocationsMatchOccupancyModel) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12DescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8);

  struct ModelPage {
    ID3D12DescriptorHeap *heap;
    std::vector<bool> occupied;
  };

  std::vector<ModelPage> pages;
  std::vector<ObjectPtr<D3D12DescriptorPool::Lease>> leases;
  const auto stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  auto release = [&](size_t i) {
    auto &lease = leases[i];
    auto &page = *std::find_if(pages.begin(), pages.end(), [&](const auto &p) { return p.heap == lease->Heap(); });
    auto offset = (lease->CPU().ptr - page.heap->GetCPUDescriptorHandleForHeapStart().ptr) / stride;
    for (size_t j = offset; j < offset + lease->Count(); ++j) {
      EXPECT_TRUE(page.occupied[j]);
      page.occupied[j] = false;
    }
    leases.erase(leases.begin() + i);
  };
  std::mt19937 random(87);
  for (int step = 0; step < 2000; ++step) {
    if (!leases.empty() && (leases.size() >= 32 || random() % 2)) {
      release(random() % leases.size());
      continue;
    }
    const uint32_t count = 1 + random() % 19;
    size_t best_page = pages.size(), best_offset = 0, best_size = SIZE_MAX;
    for (size_t i = 0; i < pages.size(); ++i) {
      const auto &bits = pages[i].occupied;
      for (size_t start = 0; start < bits.size();) {
        if (bits[start]) {
          ++start;
          continue;
        }
        size_t end = start;
        while (end < bits.size() && !bits[end])
          ++end;
        if (end - start >= count && (best_page == pages.size() || end - start < best_size)) {
          best_page = i;
          best_offset = start;
          best_size = end - start;
        }
        start = end;
      }
      if (best_page != pages.size())
        break;  // Pages are created in increasing capacity order.
    }
    auto lease = pool.Acquire(count);
    ASSERT_TRUE(lease);
    if (best_page == pages.size()) {
      uint32_t capacity = pages.empty() ? 8 : uint32_t(pages.back().occupied.size()) * 2;
      while (capacity < count)
        capacity *= 2;
      EXPECT_EQ(lease->Heap()->GetDesc().NumDescriptors, capacity);
      pages.push_back({lease->Heap(), std::vector<bool>(capacity)});
    }
    auto &page = pages[best_page];
    EXPECT_EQ(lease->Heap(), page.heap);
    const auto offset = (lease->CPU().ptr - page.heap->GetCPUDescriptorHandleForHeapStart().ptr) / stride;
    ASSERT_EQ(offset, best_offset);
    for (size_t j = offset; j < offset + count; ++j) {
      ASSERT_FALSE(page.occupied[j]);
      page.occupied[j] = true;
    }
    leases.push_back(std::move(lease));
  }
  while (!leases.empty())
    release(random() % leases.size());
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{pages.size(), 0}));
  for (const auto &page : pages) {
    auto full = pool.Acquire(uint32_t(page.occupied.size()));
    ASSERT_TRUE(full);
    EXPECT_EQ(full->Heap(), page.heap);
    EXPECT_EQ(full->CPU().ptr, page.heap->GetCPUDescriptorHandleForHeapStart().ptr);
  }
}

TEST(D3D12TransientDescriptorPool, WholePagesReuseOnlyTheirClassAndNeverOverlap) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12TransientDescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  auto first = pool.Acquire(3);
  auto second = pool.Acquire(32);
  auto large = pool.Acquire(33);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(large);
  EXPECT_EQ(first->Count(), 32u);
  EXPECT_EQ(second->Count(), 32u);
  EXPECT_EQ(large->Count(), 64u);
  EXPECT_NE(first->Heap(), second->Heap());
  EXPECT_NE(first->Heap(), large->Heap());
  EXPECT_EQ(first->Heap()->GetDesc().Flags, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
  EXPECT_NE(first->GPU().ptr, 0u);
  EXPECT_EQ(first->CPU(1).ptr - first->CPU().ptr,
            device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
  EXPECT_EQ(pool.GetPageCounts(32), (std::pair<size_t, size_t>{2, 2}));
  EXPECT_EQ(pool.GetPageCounts(64), (std::pair<size_t, size_t>{1, 1}));
  auto heap = first->Heap();
  first.Reset();
  auto recycled = pool.Acquire(1);
  ASSERT_TRUE(recycled);
  EXPECT_EQ(recycled->Heap(), heap);
  pool.Clear();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{3, 3}));
  EXPECT_FALSE(pool.Acquire(0));
  EXPECT_FALSE(pool.Acquire(1000001));
  EXPECT_FALSE(pool.Acquire(UINT32_MAX));
}

TEST(D3D12TransientDescriptorPool, RecentConcurrentDemandIsCachedThenTrimmedPerClass) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12TransientDescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  auto first = pool.Acquire(1);
  auto second = pool.Acquire(1);
  auto third = pool.Acquire(1);
  auto other = pool.Acquire(64);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(third);
  ASSERT_TRUE(other);
  first.Reset();
  second.Reset();
  third.Reset();
  EXPECT_EQ(pool.GetPageCounts(32), (std::pair<size_t, size_t>{3, 0}));
  first = pool.Acquire(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  second = pool.Acquire(1);
  second.Reset();
  EXPECT_EQ(pool.GetPageCounts(32), (std::pair<size_t, size_t>{2, 1}));
  EXPECT_EQ(pool.GetPageCounts(64), (std::pair<size_t, size_t>{1, 1}));
  pool.Clear();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{2, 2}));
  first.Reset();
  other.Reset();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{0, 0}));
}

TEST(D3D12TransientDescriptorPool, SamplerLimitAndConcurrentReuse) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12TransientDescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
  auto last = pool.Acquire(1025);
  ASSERT_TRUE(last);
  EXPECT_EQ(last->Count(), 2048u);
  EXPECT_FALSE(pool.Acquire(2049));
  std::array<std::thread, 4> threads;
  for (auto &thread : threads)
    thread = std::thread([&] {
      for (int i = 0; i < 100; ++i) {
        auto page = pool.Acquire(32);
        ASSERT_TRUE(page);
        EXPECT_EQ(page->Count(), 32u);
      }
    });
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(pool.GetPageCounts(32).second, 0u);
  EXPECT_LE(pool.GetPageCounts(32).first, 4u);
  EXPECT_EQ(pool.GetPageCounts(2048), (std::pair<size_t, size_t>{1, 1}));
}

}  // namespace grassland::rhi::backend::d3d12
#endif

#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/rhi.h"

namespace grassland::rhi {
TEST(VulkanDescriptorPool, GroupsReleaseSetsAndTrimIdlePages) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto native = static_cast<backend::vulkan::VulkanDevice *>(device.Get());
  BindingLayoutDesc layout{{{0, BINDING_UNIFORM_BUFFER}}};
  std::vector<ObjPtr<IBindingGroup>> groups;
  for (int i = 0; i < 130; ++i) {
    groups.emplace_back(device->CreateBindingGroup(layout));
    ASSERT_TRUE(groups.back());
  }
  EXPECT_EQ(native->DescriptorPool().GetPageCounts(), (std::pair<size_t, size_t>{3, 3}));
  for (int i = 64; i < 128; ++i)
    groups[i].Release();
  EXPECT_EQ(native->DescriptorPool().GetPageCounts(), (std::pair<size_t, size_t>{3, 2}));
  groups.clear();
  EXPECT_EQ(native->DescriptorPool().GetPageCounts(), (std::pair<size_t, size_t>{1, 0}));
  ObjPtr<IBindingGroup> reused = device->CreateBindingGroup(layout);
  ASSERT_TRUE(reused);
  EXPECT_EQ(native->DescriptorPool().GetPageCounts(), (std::pair<size_t, size_t>{1, 1}));
}

TEST(VulkanDescriptorPool, EqualNativeTypeCountsSharePagesAndDifferentPatternsStaySeparate) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto native = static_cast<backend::vulkan::VulkanDevice *>(device.Get());
  BindingLayoutDesc array{{{0, BINDING_STRUCTURED_BUFFER, 2, SHADER_VISIBILITY_VERTEX, 16}}};
  BindingLayoutDesc split{{{7, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 16},
                           {3, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 16}}};
  BindingLayoutDesc single{{{0, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_VERTEX, 16}}};
  std::vector<ObjPtr<IBindingGroup>> groups;
  for (int i = 0; i < 65; ++i) {
    if (i % 2)
      groups.emplace_back(device->CreateBindingGroup(split));
    else
      groups.emplace_back(device->CreateBindingGroup(array));
    ASSERT_TRUE(groups.back());
  }
  auto &pool = native->DescriptorPool();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{2, 2}));
  ObjPtr<IBindingGroup> other = device->CreateBindingGroup(single);
  ASSERT_TRUE(other);
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{3, 3}));
  // Reuse holes without resetting a page whose other sets are still alive.
  for (int i = 0; i < 64; i += 2)
    groups[i].Release();
  for (int i = 0; i < 64; i += 2) {
    groups[i] = device->CreateBindingGroup(split);
    ASSERT_TRUE(groups[i]);
  }
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{3, 3}));
  groups.clear();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{2, 1}));
  other.Release();
  EXPECT_EQ(pool.GetPageCounts(), (std::pair<size_t, size_t>{2, 0}));
}

TEST(VulkanDescriptorPool, LargeDescriptorArraysUseFullPageCapacity) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto native = static_cast<backend::vulkan::VulkanDevice *>(device.Get());
  BindingLayoutDesc layout{{{0, BINDING_STRUCTURED_BUFFER, 1025, SHADER_VISIBILITY_FRAGMENT, 16}}};
  // No library-imposed 1024 limit; hardware limits remain authoritative.
  ASSERT_TRUE(backend::ValidateBindingLayout(layout));
  if (!backend::vulkan::ValidatePipelineBindings(native->NativePhysicalDevice(), {{layout}}))
    GTEST_SKIP() << "Device does not support 1025 storage descriptors in one set/stage.";
  std::vector<ObjPtr<IBindingGroup>> groups;
  for (int i = 0; i < 65; ++i) {
    groups.emplace_back(device->CreateBindingGroup(layout));
    ASSERT_TRUE(groups.back());
    EXPECT_EQ(native->DescriptorPool().GetPageCounts().first, size_t(i / 64 + 1));
  }
  groups.clear();
  EXPECT_EQ(native->DescriptorPool().GetPageCounts(), (std::pair<size_t, size_t>{1, 0}));
}
}  // namespace grassland::rhi
#endif
