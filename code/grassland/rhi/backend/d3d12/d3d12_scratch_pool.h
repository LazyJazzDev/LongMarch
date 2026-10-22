#pragma once
#include <chrono>
#include <deque>
#include <mutex>

#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

namespace grassland::rhi::backend::d3d12 {
// All users must submit to the same physical queue and synchronize scratch accesses.
// Contexts retain the returned storage directly, including buffers replaced by resizing.
class D3D12ScratchPool {
 public:
  explicit D3D12ScratchPool(D3D12Device *device);
  // Share the current buffer; resize on demand using the last second's peak request.
  ObjectPtr<D3D12AllocatedBuffer> Acquire(uint64_t size);
  void Clear();

 private:
  D3D12Device *device_;
  std::mutex mutex_;
  ObjectPtr<D3D12AllocatedBuffer> buffer_;
  std::deque<std::pair<uint64_t, std::chrono::steady_clock::time_point>> peaks_;
};
}  // namespace grassland::rhi::backend::d3d12
