#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanBuffer::VulkanBuffer(VulkanDevice *device, const BufferDesc &desc) : Buffer(desc), device_(device) {
}

Device *VulkanBuffer::Owner() const {
  return device_;
}

}  // namespace grassland::rhi::backend::vulkan
