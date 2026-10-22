#pragma once
#include <chrono>
#include <deque>
#include <mutex>

#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"

namespace grassland::rhi::backend::vulkan {
// All users must submit to the same physical queue and synchronize scratch accesses.
// Contexts retain the returned storage directly, including buffers replaced by resizing.
class VulkanScratchPool {
 public:
  explicit VulkanScratchPool(VulkanDevice *device);
  // Share the current buffer; resize on demand using the last second's peak request.
  ObjectPtr<VulkanAllocatedBuffer> Acquire(uint64_t size);
  void Clear();

 private:
  VulkanDevice *device_;
  std::mutex mutex_;
  ObjectPtr<VulkanAllocatedBuffer> buffer_;
  std::deque<std::pair<uint64_t, std::chrono::steady_clock::time_point>> peaks_;
};
}  // namespace grassland::rhi::backend::vulkan
