#include "grassland/rhi/backend/vulkan/vulkan_upload_pool.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanUploadPool::VulkanUploadPool(VulkanDevice *device) : device_(device) {
}

ObjectPtr<VulkanAllocatedBuffer> VulkanUploadPool::Acquire(uint64_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = available_.begin(); it != available_.end(); ++it) {
    if ((*it)->GetDesc().size >= size) {
      auto result = std::move(*it);
      available_.erase(it);
      return result;
    }
  }
  uint64_t capacity = 65536;
  while (capacity < size && capacity <= UINT64_MAX / 2)
    capacity *= 2;
  if (capacity < size)
    return {};
  auto result = ObjectPtr<VulkanAllocatedBuffer>::Adopt(new VulkanAllocatedBuffer(device_, {capacity}));
  if (!result->Initialize(true))
    return {};
  return result;
}

void VulkanUploadPool::Release(ObjectPtr<VulkanAllocatedBuffer> buffer) {
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    available_.push_back(std::move(buffer));
  } catch (const std::bad_alloc &) {
  }
}

void VulkanUploadPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  available_.clear();
}
}  // namespace grassland::rhi::backend::vulkan
