#pragma once
#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {
VkResult SingleTimeCommand(VkDevice device,
                           VkQueue queue,
                           VkCommandPool pool,
                           const std::function<void(VkCommandBuffer)> &function);
VkResult SingleTimeCommand(VkDevice device,
                           VkQueue queue,
                           VkCommandPool pool,
                           const std::function<void(VkCommandBuffer)> &function,
                           VkSubmitInfo &submit_info);
}  // namespace grassland::graphics::backend::vulkan
