#pragma once
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_upload_buffer_lease.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12UploadPool {
 public:
  explicit D3D12UploadPool(D3D12Device *device);
  ObjectPtr<D3D12UploadBufferLease> Acquire(uint64_t size);
  // Called by the lease destructor; callers release the lease reference instead.
  void ReleaseBuffer(D3D12AllocatedBuffer *buffer);
  // Allocated / occupied physical buffers in the requested size class.
  std::pair<size_t, size_t> GetBufferCounts(uint64_t size) const;
  void Clear();

 private:
  D3D12Device *device_;
  mutable std::mutex mutex_;

  struct Bucket {
    void Record(size_t occupancy, std::chrono::steady_clock::time_point now);
    std::unordered_map<D3D12AllocatedBuffer *, ObjectPtr<D3D12AllocatedBuffer>> buffers;
    std::vector<D3D12AllocatedBuffer *> available;
    size_t allocated{}, occupied{};
    std::deque<std::pair<size_t, std::chrono::steady_clock::time_point>> peaks;
  };

  // 64 KiB through 2^63 bytes, indexed directly by the requested power-of-two class.
  std::array<Bucket, 48> buckets_;
};
}  // namespace grassland::rhi::backend::d3d12
