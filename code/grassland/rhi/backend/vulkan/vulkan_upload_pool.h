#pragma once
#include <mutex>

#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"

namespace grassland::rhi::backend::vulkan {
class VulkanUploadPool {
 public:
  explicit VulkanUploadPool(VulkanDevice *device);
  ObjectPtr<VulkanAllocatedBuffer> Acquire(uint64_t size);
  void Release(ObjectPtr<VulkanAllocatedBuffer> buffer);
  void Clear();

 private:
  VulkanDevice *device_;
  std::mutex mutex_;
  std::vector<ObjectPtr<VulkanAllocatedBuffer>> available_;
};
}  // namespace grassland::rhi::backend::vulkan
