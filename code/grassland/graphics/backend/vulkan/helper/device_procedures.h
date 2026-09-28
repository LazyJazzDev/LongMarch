#pragma once
#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {
struct DeviceProcedures {
  /** Ray Tracing Procedures */
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdBuildAccelerationStructuresKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateAccelerationStructureKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetAccelerationStructureBuildSizesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetBufferDeviceAddressKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkDestroyAccelerationStructureKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetAccelerationStructureDeviceAddressKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateRayTracingPipelinesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetRayTracingShaderGroupHandlesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdTraceRaysKHR);
};

void LoadRayTracingProcedures(VkDevice device, DeviceProcedures &out);
void LoadAccelerationStructureProcedures(VkDevice device, DeviceProcedures &out);
}  // namespace grassland::graphics::backend::vulkan
