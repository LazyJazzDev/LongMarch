#pragma once

#include "grassland/vulkan/vulkan_util.h"

namespace grassland::vulkan {
class InstanceProcedures {
 public:
  InstanceProcedures() = default;
  void Initialize(VkInstance instance, bool enabled_validation_layers);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateDebugUtilsMessengerEXT);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkDestroyDebugUtilsMessengerEXT);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkSetDebugUtilsObjectNameEXT);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdBeginRenderingKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdEndRenderingKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdSetPrimitiveTopologyEXT);
};
}  // namespace grassland::vulkan
