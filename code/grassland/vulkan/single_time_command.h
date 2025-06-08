#pragma once
#include "grassland/vulkan/command_pool.h"
#include "grassland/vulkan/queue.h"

namespace grassland::vulkan {
VkResult SingleTimeCommand(const Queue *queue,
                           const CommandPool *command_pool,
                           std::function<void(VkCommandBuffer)> function);

VkResult SingleTimeCommand(const Queue *queue,
                           const CommandPool *command_pool,
                           std::function<void(VkCommandBuffer)> function,
                           VkSubmitInfo &submit_info);
}  // namespace grassland::vulkan
