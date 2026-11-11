#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
// Device-specific entry points. Unsupported feature groups remain null.
struct DeviceProcedures {
  bool Load(VkDevice device, uint32_t api_version, const DeviceFeatures &enabled);

  PFN_vkCmdBeginRendering CmdBeginRendering{};
  PFN_vkCmdEndRendering CmdEndRendering{};
  PFN_vkCreateAccelerationStructureKHR CreateAccelerationStructureKHR{};
  PFN_vkDestroyAccelerationStructureKHR DestroyAccelerationStructureKHR{};
  PFN_vkGetAccelerationStructureBuildSizesKHR GetAccelerationStructureBuildSizesKHR{};
  PFN_vkGetAccelerationStructureDeviceAddressKHR GetAccelerationStructureDeviceAddressKHR{};
  PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAccelerationStructuresKHR{};
  PFN_vkCmdWriteAccelerationStructuresPropertiesKHR CmdWriteAccelerationStructuresPropertiesKHR{};
  PFN_vkCmdCopyAccelerationStructureKHR CmdCopyAccelerationStructureKHR{};
  PFN_vkCreateRayTracingPipelinesKHR CreateRayTracingPipelinesKHR{};
  PFN_vkGetRayTracingShaderGroupHandlesKHR GetRayTracingShaderGroupHandlesKHR{};
  PFN_vkCmdTraceRaysKHR CmdTraceRaysKHR{};
};
}  // namespace grassland::rhi::backend::vulkan
