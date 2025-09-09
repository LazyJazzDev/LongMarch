#pragma once
#include "grassland/vulkan/vulkan_util.h"

namespace CD::vulkan {
class DeviceProcedures {
 public:
  explicit DeviceProcedures();
  void GetRayTracingProcedures(VkDevice device);

  /** Ray Tracing Procedures */
  CD_VULKAN_PROCEDURE_VAR(vkCmdBuildAccelerationStructuresKHR);
  CD_VULKAN_PROCEDURE_VAR(vkCreateAccelerationStructureKHR);
  CD_VULKAN_PROCEDURE_VAR(vkGetAccelerationStructureBuildSizesKHR);
  CD_VULKAN_PROCEDURE_VAR(vkGetBufferDeviceAddressKHR);
  CD_VULKAN_PROCEDURE_VAR(vkDestroyAccelerationStructureKHR);
  CD_VULKAN_PROCEDURE_VAR(vkGetAccelerationStructureDeviceAddressKHR);
  CD_VULKAN_PROCEDURE_VAR(vkCreateRayTracingPipelinesKHR);
  CD_VULKAN_PROCEDURE_VAR(vkGetRayTracingShaderGroupHandlesKHR);
  CD_VULKAN_PROCEDURE_VAR(vkCmdTraceRaysKHR);
};
}  // namespace CD::vulkan
