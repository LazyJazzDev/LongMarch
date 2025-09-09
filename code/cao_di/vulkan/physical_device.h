#pragma once

#include "cao_di/vulkan/instance.h"

namespace CD::vulkan {

// Physical Device
class PhysicalDevice {
 public:
  explicit PhysicalDevice(VkPhysicalDevice physical_device);

  ~PhysicalDevice();

  VkPhysicalDevice Handle() const;

  // Functions that Return Key Features and Properties
  VkPhysicalDeviceFeatures GetPhysicalDeviceFeatures() const;

  VkPhysicalDeviceProperties GetPhysicalDeviceProperties() const;

  VkPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties() const;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR GetPhysicalDeviceRayTracingPipelineProperties() const;

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR GetPhysicalDeviceRayTracingPipelineFeatures() const;

  uint64_t GetDeviceLocalMemorySize() const;

  std::vector<VkExtensionProperties> GetDeviceExtensions() const;

  std::vector<VkQueueFamilyProperties> GetQueueFamilyProperties() const;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR GetRayTracingProperties() const;

  bool IsExtensionSupported(const char *extension_name) const;

  bool SupportGeometryShader() const;

  bool SupportRayTracing() const;

  uint64_t Evaluate() const;

  uint32_t GraphicsFamilyIndex() const;

  uint32_t PresentFamilyIndex(const Surface *surface) const;

  uint32_t ComputeFamilyIndex() const;

  uint32_t TransferFamilyIndex() const;

  VkSampleCountFlagBits GetMaxUsableSampleCount() const;

  bool CheckFeatureSupport(const struct DeviceFeatureRequirement &feature_requirement) const;

#if defined(CHANGZHENG_CUDA_RUNTIME)
  int GetCUDADeviceIndex() const;
#endif

 private:
  VkPhysicalDevice physical_device_{};
};

}  // namespace CD::vulkan
