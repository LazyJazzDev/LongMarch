#pragma once
#include <chrono>
#include <deque>
#include <mutex>

#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"

namespace grassland::rhi::backend::metal {
// All users must submit to the same physical queue and synchronize scratch accesses.
// Contexts retain the returned storage directly, including buffers replaced by resizing.
class MetalScratchPool {
 public:
  explicit MetalScratchPool(MetalDevice *device);
  // Share the current buffer; resize on demand using the last second's peak request.
  ObjectPtr<MetalAllocatedBuffer> Acquire(uint64_t size);
  void Clear();

 private:
  MetalDevice *device_;
  std::mutex mutex_;
  ObjectPtr<MetalAllocatedBuffer> buffer_;
  std::deque<std::pair<uint64_t, std::chrono::steady_clock::time_point>> peaks_;
};
}  // namespace grassland::rhi::backend::metal
