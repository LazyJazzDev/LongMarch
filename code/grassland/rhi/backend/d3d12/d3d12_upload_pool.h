#pragma once
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12UploadPool {
 public:
  // Borrows pool-owned storage. The pool and its Device must outlive every lease.
  class Lease final : public D3D12Buffer {
   public:
    Lease(D3D12UploadPool *pool, D3D12AllocatedBuffer *buffer, size_t level);
    ~Lease() override;
    BufferDesc GetDesc() const override;
    ID3D12Resource *NativeBuffer() const override;
    D3D12_RESOURCE_STATES DefaultState() const override;
    bool Write(const void *data, uint64_t size, uint64_t offset = 0);

   private:
    D3D12UploadPool *pool_;
    D3D12AllocatedBuffer *buffer_;
    size_t level_;
  };

  explicit D3D12UploadPool(D3D12Device *device);
  ObjectPtr<Lease> Acquire(uint64_t size);
  // Allocated / occupied physical buffers in the requested size class.
  std::pair<size_t, size_t> GetBufferCounts(uint64_t size) const;
  void Clear();

 private:
  // Called by the lease destructor; callers release the lease reference instead.
  void ReleaseBuffer(D3D12AllocatedBuffer *buffer, size_t level);

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
