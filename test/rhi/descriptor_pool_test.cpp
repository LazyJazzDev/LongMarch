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
}  // namespace grassland::rhi::backend::d3d12
#endif
