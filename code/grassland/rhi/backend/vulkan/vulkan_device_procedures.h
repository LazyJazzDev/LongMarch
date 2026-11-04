#pragma once
#if defined(LONGMARCH_CUDA_RUNTIME) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "grassland/rhi/backend/vulkan/vulkan_util.h"

#if defined(LONGMARCH_CUDA_RUNTIME) && defined(_WIN32)
// Windows types must be declared before the Vulkan Win32 extension entry points.
#include <vulkan/vulkan_win32.h>
#endif

namespace grassland::rhi::backend::vulkan {
// Device-specific entry points. Unsupported feature groups remain null.
struct DeviceProcedures {
  bool Load(VkDevice device, uint32_t api_version, const DeviceFeatures &enabled);

#if defined(LONGMARCH_CUDA_RUNTIME)
#if defined(_WIN32)
  PFN_vkGetMemoryWin32HandleKHR GetMemoryWin32HandleKHR{};
#else
  PFN_vkGetMemoryFdKHR GetMemoryFdKHR{};
#endif
#endif
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
