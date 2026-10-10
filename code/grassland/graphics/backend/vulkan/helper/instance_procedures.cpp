#include "grassland/graphics/backend/vulkan/helper/instance_procedures.h"

namespace grassland::graphics::backend::vulkan {

namespace {
template <class FuncTy>
FuncTy GetProcedure(VkInstance instance, const char *function_name) {
  auto func = (FuncTy)vkGetInstanceProcAddr(instance, function_name);
  if (!func) {
    Warning("Failed to load instance function: {}", function_name);
  }
  return func;
};

template <class FuncTy>
FuncTy GetDeviceRenderingProcedure(VkDevice device, const char *extension_name, const char *core_name) {
  auto func = reinterpret_cast<FuncTy>(vkGetDeviceProcAddr(device, extension_name));
  if (!func) {
    func = reinterpret_cast<FuncTy>(vkGetDeviceProcAddr(device, core_name));
  }
  if (!func) {
    throw std::runtime_error(std::string("Missing Vulkan rendering function: ") + extension_name);
  }
  return func;
}
}  // namespace

#define GET_PROCEDURE(instance, function_name) \
  out.function_name = grassland::graphics::backend::vulkan::GetProcedure<PFN_##function_name>(instance, #function_name)

void LoadInstanceProcedures(VkInstance instance, bool enabled_validation_layers, InstanceProcedures &out) {
  if (enabled_validation_layers) {
    GET_PROCEDURE(instance, vkCreateDebugUtilsMessengerEXT);
    GET_PROCEDURE(instance, vkDestroyDebugUtilsMessengerEXT);
    GET_PROCEDURE(instance, vkSetDebugUtilsObjectNameEXT);
  }
}

void LoadDeviceRenderingProcedures(VkDevice device, InstanceProcedures &out) {
  // Resolve device extensions after device creation. Some loaders (including
  // the HarmonyOS emulator) do not expose these through vkGetInstanceProcAddr.
  out.vkCmdBeginRenderingKHR =
      GetDeviceRenderingProcedure<PFN_vkCmdBeginRenderingKHR>(device, "vkCmdBeginRenderingKHR", "vkCmdBeginRendering");
  out.vkCmdEndRenderingKHR =
      GetDeviceRenderingProcedure<PFN_vkCmdEndRenderingKHR>(device, "vkCmdEndRenderingKHR", "vkCmdEndRendering");
  out.vkCmdSetPrimitiveTopologyEXT = GetDeviceRenderingProcedure<PFN_vkCmdSetPrimitiveTopologyEXT>(
      device, "vkCmdSetPrimitiveTopologyEXT", "vkCmdSetPrimitiveTopology");
}
}  // namespace grassland::graphics::backend::vulkan
