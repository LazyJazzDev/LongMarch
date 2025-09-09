#include "cao_di/vulkan/device_procedures.h"

namespace CD::vulkan {

namespace {
template <class FuncTy>
FuncTy GetProcedure(VkDevice device, const char *function_name) {
  auto func = reinterpret_cast<FuncTy>(vkGetDeviceProcAddr(device, function_name));
  if (!func) {
    ThrowError("Failed to load device function: {}", function_name);
  }
  return func;
};
}  // namespace

#define GET_PROCEDURE(device, function_name) \
  function_name = CD::vulkan::GetProcedure<PFN_##function_name>(device, #function_name)

DeviceProcedures::DeviceProcedures() = default;

void DeviceProcedures::GetRayTracingProcedures(VkDevice device) {
  GET_PROCEDURE(device, vkGetBufferDeviceAddressKHR);
  GET_PROCEDURE(device, vkGetAccelerationStructureBuildSizesKHR);
  GET_PROCEDURE(device, vkCreateAccelerationStructureKHR);
  GET_PROCEDURE(device, vkCmdBuildAccelerationStructuresKHR);
  GET_PROCEDURE(device, vkDestroyAccelerationStructureKHR);
  GET_PROCEDURE(device, vkGetAccelerationStructureDeviceAddressKHR);
  GET_PROCEDURE(device, vkCreateRayTracingPipelinesKHR);
  GET_PROCEDURE(device, vkGetRayTracingShaderGroupHandlesKHR);
  GET_PROCEDURE(device, vkCmdTraceRaysKHR);
}
}  // namespace CD::vulkan
