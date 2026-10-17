#pragma once
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend::vulkan {

bool Check(VkResult result, const char *operation);

template <typename T, typename Enumerate>
bool EnumerateList(Enumerate enumerate, std::vector<T> &values) {
  values.clear();
  VkResult result;
  do {
    uint32_t count = 0;
    if (!Check(enumerate(&count, nullptr), "Vulkan enumeration count")) {
      values.clear();
      return false;
    }
    values.resize(count);
    if (!count)
      return true;
    result = enumerate(&count, values.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
      Check(result, "Vulkan enumeration");
      values.clear();
      return false;
    }
    values.resize(count);
  } while (result == VK_INCOMPLETE);
  return true;
}

// Rebuild the chain in place; its pointers must never be copied to another object.
class DeviceFeatures {
 public:
  DeviceFeatures(VkPhysicalDevice device, uint32_t api_version);
  DeviceFeatures(const DeviceFeatures &) = delete;
  DeviceFeatures &operator=(const DeviceFeatures &) = delete;

  explicit operator bool() const noexcept;

  const VkPhysicalDeviceFeatures2 &Features() const noexcept;

  const std::vector<const char *> &Extensions() const noexcept;

  bool RayTracingPipeline() const;

  bool RayQuery() const;
  bool TimelineSemaphore() const;

 private:
  bool valid_{false};
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
