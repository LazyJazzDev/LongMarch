#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"

namespace grassland::rhi::backend::vulkan {

VulkanQueue::VulkanQueue(VulkanDevice *device, QueueType type) : IQueue(device, type) {
}

std::unique_ptr<ICommandContext> VulkanQueue::CreateCommandContext() const {
  return std::make_unique<VulkanCommandContext>(static_cast<VulkanDevice *>(Owner()), Type());
}

}  // namespace grassland::rhi::backend::vulkan
