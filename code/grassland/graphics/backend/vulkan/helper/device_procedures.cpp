#include "grassland/graphics/backend/vulkan/helper/device_procedures.h"

namespace grassland::graphics::backend::vulkan {

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
  out.function_name = grassland::graphics::backend::vulkan::GetProcedure<PFN_##function_name>(device, #function_name)

void LoadAccelerationStructureProcedures(VkDevice device, DeviceProcedures &out) {
  GET_PROCEDURE(device, vkGetBufferDeviceAddressKHR);
  GET_PROCEDURE(device, vkGetAccelerationStructureBuildSizesKHR);
  GET_PROCEDURE(device, vkCreateAccelerationStructureKHR);
  GET_PROCEDURE(device, vkCmdBuildAccelerationStructuresKHR);
  GET_PROCEDURE(device, vkDestroyAccelerationStructureKHR);
  GET_PROCEDURE(device, vkGetAccelerationStructureDeviceAddressKHR);
}

void LoadRayTracingProcedures(VkDevice device, DeviceProcedures &out) {
  LoadAccelerationStructureProcedures(device, out);
  GET_PROCEDURE(device, vkCreateRayTracingPipelinesKHR);
  GET_PROCEDURE(device, vkGetRayTracingShaderGroupHandlesKHR);
  GET_PROCEDURE(device, vkCmdTraceRaysKHR);
}
}  // namespace grassland::graphics::backend::vulkan
