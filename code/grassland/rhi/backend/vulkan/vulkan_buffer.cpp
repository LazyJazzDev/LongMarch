#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanBuffer::VulkanBuffer(VulkanDevice *device, const BufferDesc &desc) : Buffer(device, desc) {
}
}  // namespace grassland::rhi::backend::vulkan
