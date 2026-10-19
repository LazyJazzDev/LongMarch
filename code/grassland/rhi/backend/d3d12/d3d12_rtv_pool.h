#pragma once
#include <mutex>
#include <vector>

#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
// Device-owned pages. A context holds slot leases until its last reference is released.
// Slots borrow their parent pool; the Device drains submitted contexts before teardown.
class D3D12RTVPool {
 public:
  explicit D3D12RTVPool(ID3D12Device *device);
  bool Acquire(ID3D12Resource *image, uint64_t &slot, D3D12_CPU_DESCRIPTOR_HANDLE &handle);
  void Release(uint64_t slot);

 private:
  struct Page {
    ComPtr<ID3D12DescriptorHeap> heap;
    uint64_t occupied{};
  };

  ID3D12Device *device_;
  UINT stride_;
  std::mutex mutex_;
  std::vector<Page> pages_;
};
}  // namespace grassland::rhi::backend::d3d12
