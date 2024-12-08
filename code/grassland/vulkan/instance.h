#pragma once

#include "grassland/vulkan/instance_procedures.h"
#include "grassland/vulkan/surface.h"
#include "grassland/vulkan/vulkan_util.h"

namespace grassland::vulkan {

struct InstanceCreateHint {
  bool enable_validation_layers{kDefaultEnableValidationLayers};

  std::vector<const char *> extensions{};

  VkApplicationInfo app_info{};

  explicit InstanceCreateHint();

  void SetValidationLayersEnabled(
      bool enabled = kDefaultEnableValidationLayers);

  void AddExtension(const char *extension);

 private:
  void ApplyGLFWSurfaceSupport();
};

class Instance {
 public:
  Instance(InstanceCreateHint create_hint,
           VkInstance instance,
           VkDebugUtilsMessengerEXT debug_messenger,
           InstanceProcedures instance_procedures);

  ~Instance();

  operator VkInstance() const {
    return instance_;
  }

  VkInstance Handle() const {
    return instance_;
  }

  const InstanceCreateHint &CreateHint() const {
    return create_hint_;
  }

  VkResult CreateSurfaceFromGLFWWindow(GLFWwindow *window,
                                       double_ptr<Surface> pp_surface) const;

  std::vector<class PhysicalDevice> EnumeratePhysicalDevices() const;

  VkResult CreateDevice(
      const PhysicalDevice &physical_device,
      const struct DeviceFeatureRequirement &device_feature_requirement,
      struct DeviceCreateInfo create_info,
      double_ptr<struct Device> pp_device) const;

  VkResult CreateDevice(Surface *surface,
                        bool enable_raytracing_extension,
                        int device_index,
                        double_ptr<struct Device> pp_device) const;

  VkResult CreateDevice(
      const struct DeviceFeatureRequirement &device_feature_requirement,
      int device_index,
      double_ptr<struct Device> pp_device) const;

  VkResult CreateDevice(Surface *surface,
                        bool enable_raytracing_extension,
                        double_ptr<struct Device> pp_device) const;

  VkResult CreateDevice(
      const struct DeviceFeatureRequirement &device_feature_requirement,
      double_ptr<struct Device> pp_device) const;

  InstanceProcedures Procedures() const {
    return instance_procedures_;
  }

 private:
  InstanceCreateHint create_hint_;

  VkInstance instance_{};
  VkDebugUtilsMessengerEXT debug_messenger_{};
  InstanceProcedures instance_procedures_{};
};

VkResult CreateInstance(
    InstanceCreateHint create_hint,
    double_ptr<Instance> pp_instance = static_cast<Instance **>(nullptr));

}  // namespace grassland::vulkan
