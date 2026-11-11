#pragma once
#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
// The Device owns the pool; leases borrow their exact page and range.
// Pages remain allocated until pool destruction.
class D3D12DescriptorPool final {
  class LeaseImpl;

  struct Page {
    ComPtr<ID3D12DescriptorHeap> heap;
    uint32_t capacity{};
    size_t occupied{};
    std::map<uint32_t, uint32_t> ranges;                // offset -> count, for neighboring ranges
    std::set<std::pair<uint32_t, uint32_t>> available;  // (count, offset), for best fit
  };

 public:
  class ILease : public IObject {
   public:
    virtual ID3D12DescriptorHeap *Heap() const = 0;
    virtual uint32_t Count() const = 0;
    virtual D3D12_CPU_DESCRIPTOR_HANDLE CPU(uint32_t index = 0) const = 0;
  };

  using Lease = ILease;

  D3D12DescriptorPool(::ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t page_capacity = 256);
  ObjectPtr<Lease> Acquire(uint32_t count);
  std::pair<size_t, size_t> GetPageCounts() const;

 private:
  void ReleaseRange(Page *page, uint32_t offset, uint32_t count);
  ::ID3D12Device *device_;
  D3D12_DESCRIPTOR_HEAP_TYPE type_;
  uint32_t page_capacity_, stride_;
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Page>> pages_;
};
}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::D3D12DescriptorPool::ILease)
