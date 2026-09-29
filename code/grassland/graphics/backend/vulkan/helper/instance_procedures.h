#pragma once

#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {
struct InstanceProcedures {
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateDebugUtilsMessengerEXT);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkDestroyDebugUtilsMessengerEXT);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkSetDebugUtilsObjectNameEXT);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdBeginRenderingKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdEndRenderingKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdSetPrimitiveTopologyEXT);
};

void LoadInstanceProcedures(VkInstance instance, bool enabled_validation_layers, InstanceProcedures &out);
}  // namespace grassland::graphics::backend::vulkan
