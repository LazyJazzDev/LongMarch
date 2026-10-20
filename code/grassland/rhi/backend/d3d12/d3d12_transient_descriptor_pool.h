#pragma once
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
// Device-owned, shader-visible whole heaps. A context retains each lease until
// its recorded commands are discarded or GPU work has completed.
class D3D12TransientDescriptorPool {
 public:
  class Lease final : public Object<IObject> {
   public:
    Lease(D3D12TransientDescriptorPool *pool, ID3D12DescriptorHeap *heap, size_t level);
    ~Lease() override;
    ID3D12DescriptorHeap *Heap() const;
    uint32_t Count() const;
    D3D12_CPU_DESCRIPTOR_HANDLE CPU(uint32_t index = 0) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GPU(uint32_t index = 0) const;

   private:
    D3D12TransientDescriptorPool *pool_;
    ID3D12DescriptorHeap *heap_;
    size_t level_;
    uint32_t count_;
  };

  D3D12TransientDescriptorPool(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type);
  // Round to a power-of-two class, with a final class at the native heap limit.
  ObjectPtr<Lease> Acquire(uint32_t minimum_count);
  // Allocated / occupied pages, in one class or across all classes (count == 0).
  std::pair<size_t, size_t> GetPageCounts(uint32_t minimum_count = 0) const;
  void Clear();

 private:
  void ReleasePage(ID3D12DescriptorHeap *heap, size_t level);
  ID3D12Device *device_;
  D3D12_DESCRIPTOR_HEAP_TYPE type_;
  uint32_t limit_, stride_;
  mutable std::mutex mutex_;

  struct Bucket {
    void Record(size_t occupancy, std::chrono::steady_clock::time_point now);
    std::unordered_map<ID3D12DescriptorHeap *, ComPtr<ID3D12DescriptorHeap>> pages;
    std::vector<ID3D12DescriptorHeap *> available;
    size_t allocated{}, occupied{};
    std::deque<std::pair<size_t, std::chrono::steady_clock::time_point>> peaks;
  };

  // 32 through 524288 descriptors, plus the terminal 1000000-descriptor class.
  std::array<Bucket, 16> buckets_;
};
}  // namespace grassland::rhi::backend::d3d12
