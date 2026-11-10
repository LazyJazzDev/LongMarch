#pragma once
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;
VkBufferUsageFlags BufferUsage(VulkanDevice *device, bool upload = false, bool download = false);

// Resource access only. Derived classes define allocation ownership (standalone or future pool leases).
class VulkanBuffer : public Buffer {
 public:
  virtual VkBuffer NativeBuffer() const = 0;

 protected:
  VulkanBuffer(VulkanDevice *device, const BufferDesc &desc);
};
}  // namespace grassland::rhi::backend::vulkan
