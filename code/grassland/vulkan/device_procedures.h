#pragma once
#include "grassland/vulkan/vulkan_util.h"

namespace grassland::vulkan {
class DeviceProcedures {
 public:
  explicit DeviceProcedures();
  void GetRayTracingProcedures(VkDevice device);

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
}  // namespace grassland::vulkan
