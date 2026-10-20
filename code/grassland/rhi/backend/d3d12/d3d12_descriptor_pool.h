#pragma once
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
// The Device owns the pool; leases borrow their exact page and range.
// Pages remain allocated until pool destruction.
class D3D12DescriptorPool {
  using FreeRanges = std::map<uint32_t, uint32_t>;  // offset -> count

  struct Page {
    ComPtr<ID3D12DescriptorHeap> heap;
    uint32_t capacity{};
    size_t id{}, occupied{};
    FreeRanges free_ranges;
  };

  // Capacity first, then best fit. Page id and offset break ties deterministically.
  using FreeKey = std::tuple<uint32_t, uint32_t, size_t, uint32_t>;
  using FreeIndex = std::map<FreeKey, Page *>;

 public:
  class Lease final : public Object<IObject> {
   public:
    Lease(D3D12DescriptorPool *pool,
          Page *page,
          uint32_t offset,
          uint32_t count,
          FreeRanges::node_type range_node,
          FreeIndex::node_type index_node);
    ~Lease() override;
    ID3D12DescriptorHeap *Heap() const;
    uint32_t Count() const;
    D3D12_CPU_DESCRIPTOR_HANDLE CPU(uint32_t index = 0) const;

   private:
    D3D12DescriptorPool *pool_;
    Page *page_;
    uint32_t offset_, count_;
    // Reserve both bookkeeping nodes on Acquire so release cannot allocate.
    FreeRanges::node_type range_node_;
    FreeIndex::node_type index_node_;
  };

  D3D12DescriptorPool(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t page_capacity = 256);
  ObjectPtr<Lease> Acquire(uint32_t count);
  std::pair<size_t, size_t> GetPageCounts() const;

 private:
  void ReleaseRange(Page *page,
                    uint32_t offset,
                    uint32_t count,
                    FreeRanges::node_type range_node,
                    FreeIndex::node_type index_node);
  ID3D12Device *device_;
  D3D12_DESCRIPTOR_HEAP_TYPE type_;
  uint32_t page_capacity_, stride_;
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Page>> pages_;
  FreeIndex free_index_;
};
}  // namespace grassland::rhi::backend::d3d12
