#pragma once
#include <deque>
#include <mutex>
#include <vector>

#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
// The Device owns the pool; leases borrow their exact page and allocation.
// Returning a later allocation never permits overwriting an earlier live range.
class D3D12DescriptorPool {
  struct Allocation {
    uint64_t end;
    bool released{};
  };

  struct Page {
    ComPtr<ID3D12DescriptorHeap> heap;
    uint32_t capacity{};
    uint64_t head{}, tail{};
    std::deque<std::unique_ptr<Allocation>> allocations;
  };

 public:
  class Lease final : public Object<IObject> {
   public:
    Lease(D3D12DescriptorPool *pool, Page *page, Allocation *allocation, uint32_t offset, uint32_t count);
    ~Lease() override;
    ID3D12DescriptorHeap *Heap() const;
    uint32_t Count() const;
    D3D12_CPU_DESCRIPTOR_HANDLE CPU(uint32_t index = 0) const;

   private:
    D3D12DescriptorPool *pool_;
    Page *page_;
    Allocation *allocation_;
    uint32_t offset_, count_;
  };

  D3D12DescriptorPool(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t page_capacity = 256);
  ObjectPtr<Lease> Acquire(uint32_t count);
  // Explicit trimming never invalidates a live lease.
  void Clear();
  std::pair<size_t, size_t> GetPageCounts() const;

 private:
  void ReleaseRange(Page *page, Allocation *allocation);
  ID3D12Device *device_;
  D3D12_DESCRIPTOR_HEAP_TYPE type_;
  uint32_t page_capacity_, stride_;
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Page>> pages_;
};
}  // namespace grassland::rhi::backend::d3d12
