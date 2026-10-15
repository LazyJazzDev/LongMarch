#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"

namespace grassland::rhi::backend::vulkan {

VulkanQueue::VulkanQueue(VulkanDevice *device, QueueType type) : Queue(device, type) {
}

std::unique_ptr<CommandContext> VulkanQueue::CreateCommandContext() const {
  auto context = std::make_unique<VulkanCommandContext>(static_cast<VulkanDevice *>(Owner()), Type());
  if (!context->Initialize())
    return {};
  return context;
}

}  // namespace grassland::rhi::backend::vulkan
