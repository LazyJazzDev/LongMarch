#pragma once
#include <chrono>
#include <deque>
#include <mutex>

#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
// All users must submit to the same physical queue and synchronize scratch accesses.
// Contexts retain the returned storage directly, including buffers replaced by resizing.
class MetalScratchPool final {
 public:
  explicit MetalScratchPool(IMetalDevice *device);
  // Share the current buffer; resize on demand using the last second's peak request.
  ObjectPtr<IMetalAllocatedBuffer> Acquire(uint64_t size);
  void Clear();

 private:
  IMetalDevice *device_;
  std::mutex mutex_;
  ObjectPtr<IMetalAllocatedBuffer> buffer_;
  uint64_t capacity_{};
  std::deque<std::pair<uint64_t, std::chrono::steady_clock::time_point>> peaks_;
};
}  // namespace grassland::rhi::backend::metal
