#pragma once
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend::vulkan {

inline void Check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}

template <typename T, typename Enumerate>
std::vector<T> EnumerateList(Enumerate enumerate) {
  std::vector<T> values;
  VkResult result;
  do {
    uint32_t count = 0;
    Check(enumerate(&count, nullptr), "Vulkan enumeration count");
    values.resize(count);
    if (!count)
      return values;
    result = enumerate(&count, values.data());
    values.resize(count);
  } while (result == VK_INCOMPLETE);
  Check(result, "Vulkan enumeration");
  return values;
}

// Rebuild the chain in place; its pointers must never be copied to another object.
class DeviceFeatures {
 public:
  DeviceFeatures(VkPhysicalDevice device, uint32_t api_version);
  DeviceFeatures(const DeviceFeatures &) = delete;
  DeviceFeatures &operator=(const DeviceFeatures &) = delete;

  const VkPhysicalDeviceFeatures2 &Features() const noexcept {
    return features_;
  }

  const std::vector<const char *> &Extensions() const noexcept {
    return extensions_;
  }

  bool RayTracingPipeline() const {
    return ray_pipeline_.rayTracingPipeline == VK_TRUE;
  }

  bool RayQuery() const {
    return ray_query_.rayQuery == VK_TRUE;
  }

 private:
  VkPhysicalDeviceFeatures2 features_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  std::vector<const char *> extensions_;
  VkPhysicalDeviceVulkan11Features v11_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features v12_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
#ifdef VK_VERSION_1_3
  VkPhysicalDeviceVulkan13Features v13_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
#endif
#ifdef VK_VERSION_1_4
  VkPhysicalDeviceVulkan14Features v14_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
#endif
  VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_pipeline_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR ray_query_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
};

}  // namespace grassland::rhi::backend::vulkan
