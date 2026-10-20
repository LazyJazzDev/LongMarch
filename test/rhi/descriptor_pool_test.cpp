#include <gtest/gtest.h>

#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {
TEST(D3D12DescriptorPool, RingRangesRetainPagesUntilOrderedReclamation) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12DescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false, 8);
  auto a = pool.Acquire(3);
  auto b = pool.Acquire(3);
  ASSERT_TRUE(a && b);
  auto heap = a->Heap();
  auto start = a->CPU().ptr;
  EXPECT_EQ(heap, b->Heap());
  EXPECT_EQ(b->CPU().ptr - start, 3 * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV));
  b = {};
  auto c = pool.Acquire(3);
  ASSERT_TRUE(c);
  EXPECT_NE(c->Heap(), heap);  // A still pins the ring head, including B's released range.
  pool.Clear();
  EXPECT_EQ(pool.GetPageCounts().first, 2u);
  a = {};
  auto d = pool.Acquire(8);
  ASSERT_TRUE(d);
  EXPECT_EQ(d->Heap(), heap);
  EXPECT_EQ(d->CPU().ptr, start);
  c = {};
  d = {};
  EXPECT_EQ(pool.GetPageCounts().first, 1u);
  EXPECT_EQ(pool.GetPageCounts().second, 0u);
  pool.Clear();
  EXPECT_EQ(pool.GetPageCounts().first, 0u);
}

TEST(D3D12DescriptorPool, WrapPaddingAndOversizedPages) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12DescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, 8);
  auto a = pool.Acquire(3);
  auto b = pool.Acquire(3);
  ASSERT_TRUE(a && b);
  auto start = a->CPU().ptr;
  a = {};
  auto c = pool.Acquire(3);  // Skip the last two slots and wrap into the released prefix.
  ASSERT_TRUE(c);
  EXPECT_EQ(c->Heap(), b->Heap());
  EXPECT_EQ(c->CPU().ptr, start);
  EXPECT_NE(c->GPU().ptr, 0u);
  auto d = pool.Acquire(1);
  ASSERT_TRUE(d);
  EXPECT_NE(d->Heap(), c->Heap());
  c = {};
  b = {};
  d = {};
  auto large = pool.Acquire(17);
  ASSERT_TRUE(large);
  EXPECT_EQ(large->Count(), 17u);
  EXPECT_EQ(pool.GetPageCounts().first, 2u);
  large = {};
  EXPECT_EQ(pool.GetPageCounts().first, 1u);
  EXPECT_FALSE(pool.Acquire(0));
  EXPECT_FALSE(pool.Acquire(UINT32_MAX));
}

TEST(D3D12DescriptorPool, PageReservationsDoNotOverlapAndGrowForLiveTables) {
  ComPtr<ID3D12Device> device;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
  D3D12DescriptorPool pool(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, 8);
  auto first = pool.AcquirePage(3);
  auto second = pool.AcquirePage(5);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->Count(), 8u);
  EXPECT_EQ(second->Count(), 8u);
  EXPECT_NE(first->Heap(), second->Heap());
  auto large = pool.AcquirePage(12);
  ASSERT_TRUE(large);
  EXPECT_GE(large->Count(), 12u);
  EXPECT_EQ(pool.GetPageCounts().second, 3u);
  auto heap = first->Heap();
  first.Reset();
  auto recycled = pool.AcquirePage(1);
  ASSERT_TRUE(recycled);
  EXPECT_EQ(recycled->Heap(), heap);
  EXPECT_FALSE(pool.AcquirePage(1000001));
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
