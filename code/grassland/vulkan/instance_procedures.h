#pragma once

#include "grassland/vulkan/vulkan_util.h"

namespace CD::vulkan {
class InstanceProcedures {
 public:
  InstanceProcedures() = default;
  void Initialize(VkInstance instance, bool enabled_validation_layers);
  CD_VULKAN_PROCEDURE_VAR(vkCreateDebugUtilsMessengerEXT);
  CD_VULKAN_PROCEDURE_VAR(vkDestroyDebugUtilsMessengerEXT);
  CD_VULKAN_PROCEDURE_VAR(vkSetDebugUtilsObjectNameEXT);
  CD_VULKAN_PROCEDURE_VAR(vkCmdBeginRenderingKHR);
  CD_VULKAN_PROCEDURE_VAR(vkCmdEndRenderingKHR);
  CD_VULKAN_PROCEDURE_VAR(vkCmdSetPrimitiveTopologyEXT);
};
}  // namespace CD::vulkan
