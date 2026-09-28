#pragma once

#include "grassland/graphics/backend/vulkan/helper/instance_procedures.h"
#include "grassland/graphics/backend/vulkan/helper/surface.h"
#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {

struct InstanceCreateHint {
  bool enable_validation_layers{kDefaultEnableValidationLayers};

  std::vector<const char *> extensions{};

  VkApplicationInfo app_info{};

  explicit InstanceCreateHint();

  void SetValidationLayersEnabled(bool enabled = kDefaultEnableValidationLayers);

  void AddExtension(const char *extension);

  bool IsEnabledExtension(const char *extension) const;

 private:
  void ApplyGLFWSurfaceSupport();
};

VkResult CreateNativeInstance(InstanceCreateHint &hint,
                              VkInstance &instance,
                              VkDebugUtilsMessengerEXT &debug_messenger,
                              InstanceProcedures &procedures);
std::vector<class PhysicalDevice> EnumerateNativePhysicalDevices(VkInstance instance);

}  // namespace grassland::graphics::backend::vulkan
