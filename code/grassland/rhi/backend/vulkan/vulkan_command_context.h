#pragma once
#include <vector>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
ICommandContext *CreateVulkanCommandContext(IVulkanQueue *queue);

}  // namespace grassland::rhi::backend::vulkan
