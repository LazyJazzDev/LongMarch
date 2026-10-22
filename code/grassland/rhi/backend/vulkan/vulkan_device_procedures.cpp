#include "grassland/rhi/backend/vulkan/vulkan_device_procedures.h"

namespace grassland::rhi::backend::vulkan {
bool DeviceProcedures::Load(VkDevice device, uint32_t api_version, const DeviceFeatures &enabled) {
  *this = {};
  if (enabled.DynamicRendering()) {
    CmdBeginRendering = reinterpret_cast<PFN_vkCmdBeginRendering>(vkGetDeviceProcAddr(
        device, api_version >= VK_API_VERSION_1_3 ? "vkCmdBeginRendering" : "vkCmdBeginRenderingKHR"));
    if (!CmdBeginRendering) {
      ReportError(-1, "Required Vulkan entry point vkCmdBeginRendering is unavailable.");
      return false;
    }
    CmdEndRendering = reinterpret_cast<PFN_vkCmdEndRendering>(
        vkGetDeviceProcAddr(device, api_version >= VK_API_VERSION_1_3 ? "vkCmdEndRendering" : "vkCmdEndRenderingKHR"));
    if (!CmdEndRendering) {
      ReportError(-1, "Required Vulkan entry point vkCmdEndRendering is unavailable.");
      return false;
    }
  }
  if (enabled.RayTracingPipeline() || enabled.RayQuery()) {
    CreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    if (!CreateAccelerationStructureKHR) {
      ReportError(-1, "Required Vulkan entry point vkCreateAccelerationStructureKHR is unavailable.");
      return false;
    }
    DestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
    if (!DestroyAccelerationStructureKHR) {
      ReportError(-1, "Required Vulkan entry point vkDestroyAccelerationStructureKHR is unavailable.");
      return false;
    }
    GetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    if (!GetAccelerationStructureBuildSizesKHR) {
      ReportError(-1, "Required Vulkan entry point vkGetAccelerationStructureBuildSizesKHR is unavailable.");
      return false;
    }
    GetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    if (!GetAccelerationStructureDeviceAddressKHR) {
      ReportError(-1, "Required Vulkan entry point vkGetAccelerationStructureDeviceAddressKHR is unavailable.");
      return false;
    }
    CmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    if (!CmdBuildAccelerationStructuresKHR) {
      ReportError(-1, "Required Vulkan entry point vkCmdBuildAccelerationStructuresKHR is unavailable.");
      return false;
    }
    CmdWriteAccelerationStructuresPropertiesKHR = reinterpret_cast<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(
        vkGetDeviceProcAddr(device, "vkCmdWriteAccelerationStructuresPropertiesKHR"));
    if (!CmdWriteAccelerationStructuresPropertiesKHR) {
      ReportError(-1, "Required Vulkan entry point vkCmdWriteAccelerationStructuresPropertiesKHR is unavailable.");
      return false;
    }
    CmdCopyAccelerationStructureKHR = reinterpret_cast<PFN_vkCmdCopyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkCmdCopyAccelerationStructureKHR"));
    if (!CmdCopyAccelerationStructureKHR) {
      ReportError(-1, "Required Vulkan entry point vkCmdCopyAccelerationStructureKHR is unavailable.");
      return false;
    }
  }
  if (enabled.RayTracingPipeline()) {
    CreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
    if (!CreateRayTracingPipelinesKHR) {
      ReportError(-1, "Required Vulkan entry point vkCreateRayTracingPipelinesKHR is unavailable.");
      return false;
    }
    GetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
        vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
    if (!GetRayTracingShaderGroupHandlesKHR) {
      ReportError(-1, "Required Vulkan entry point vkGetRayTracingShaderGroupHandlesKHR is unavailable.");
      return false;
    }
    CmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));
    if (!CmdTraceRaysKHR) {
      ReportError(-1, "Required Vulkan entry point vkCmdTraceRaysKHR is unavailable.");
      return false;
    }
  }
  return true;
}
}  // namespace grassland::rhi::backend::vulkan
