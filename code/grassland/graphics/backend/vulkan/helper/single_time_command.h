#pragma once
#include "grassland/graphics/backend/vulkan/helper/command_pool.h"
#include "grassland/graphics/backend/vulkan/helper/queue.h"

namespace grassland::graphics::backend::vulkan {
VkResult SingleTimeCommand(const Queue *queue,
                           const CommandPool *command_pool,
                           std::function<void(VkCommandBuffer)> function);

VkResult SingleTimeCommand(const Queue *queue,
                           const CommandPool *command_pool,
                           std::function<void(VkCommandBuffer)> function,
                           VkSubmitInfo &submit_info);
}  // namespace grassland::graphics::backend::vulkan
