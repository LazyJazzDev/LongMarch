#pragma once
#include "cao_di/vulkan/command_pool.h"
#include "cao_di/vulkan/queue.h"

namespace CD::vulkan {
VkResult SingleTimeCommand(const Queue *queue,
                           const CommandPool *command_pool,
                           std::function<void(VkCommandBuffer)> function);

VkResult SingleTimeCommand(const Queue *queue,
                           const CommandPool *command_pool,
                           std::function<void(VkCommandBuffer)> function,
                           VkSubmitInfo &submit_info);
}  // namespace CD::vulkan
