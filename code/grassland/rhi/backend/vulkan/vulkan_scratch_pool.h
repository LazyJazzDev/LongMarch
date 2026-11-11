#pragma once
#include <chrono>
#include <deque>
#include <mutex>

#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
// All users must submit to the same physical queue and synchronize scratch accesses.
// Contexts retain the returned storage directly, including buffers replaced by resizing.
class VulkanScratchPool final {
 public:
  explicit VulkanScratchPool(IVulkanDevice *device);
  // Share the current buffer; resize on demand using the last second's peak request.
  ObjectPtr<IVulkanAllocatedBuffer> Acquire(uint64_t size);
  void Clear();

 private:
  IVulkanDevice *device_;
  std::mutex mutex_;
  ObjectPtr<IVulkanAllocatedBuffer> buffer_;
  uint64_t capacity_{};
  std::deque<std::pair<uint64_t, std::chrono::steady_clock::time_point>> peaks_;
};
}  // namespace grassland::rhi::backend::vulkan
