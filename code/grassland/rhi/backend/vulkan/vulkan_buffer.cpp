#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VkBufferUsageFlags BufferUsage(VulkanDevice *device, bool upload, bool download) {
  VkBufferUsageFlags usage = download ? VK_BUFFER_USAGE_TRANSFER_DST_BIT
                             : upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                      : VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (device->SupportRayTracingPipeline() || device->SupportRayQuery()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if (device->SupportRayTracingPipeline())
      usage |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
  }
  return usage;
}

VulkanBuffer::VulkanBuffer(VulkanDevice *device, const BufferDesc &desc) : Buffer(device, desc) {
}
}  // namespace grassland::rhi::backend::vulkan
