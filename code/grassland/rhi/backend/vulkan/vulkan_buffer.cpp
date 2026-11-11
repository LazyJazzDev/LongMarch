#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanBuffer::VulkanBuffer(VulkanDevice *device, const BufferDesc &desc) : Buffer(desc), device_(device) {
}

Device *VulkanBuffer::Owner() const {
  return device_;
}

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanBuffer, "edd00360-77f8-46d3-9659-0c06d6616086")
