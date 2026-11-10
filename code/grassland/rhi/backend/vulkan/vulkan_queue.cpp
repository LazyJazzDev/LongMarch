#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

rhi::ICommandContext *VulkanQueue::CreateCommandContext() {
  return VulkanCommandContext::Create(this);
}

VulkanQueue::VulkanQueue(VulkanDevice *device, QueueType type) : Queue(device, type) {
}

}  // namespace grassland::rhi::backend::vulkan
