#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

DeviceFeatures::DeviceFeatures(VkPhysicalDevice device, uint32_t api_version) {
  std::vector<VkExtensionProperties> available;
  if (!EnumerateList(
          [device](uint32_t *count, VkExtensionProperties *data) {
            return vkEnumerateDeviceExtensionProperties(device, nullptr, count, data);
          },
          available))
    return;
  const auto has = [&](const char *name) {
    return std::any_of(available.begin(), available.end(),
                       [&](const auto &entry) { return std::strcmp(entry.extensionName, name) == 0; });
  };
  features_.pNext = &v11_;
  v11_.pNext = &v12_;
  void **tail = &v12_.pNext;
#ifdef VK_VERSION_1_3
  if (api_version >= VK_API_VERSION_1_3) {
    *tail = &v13_;
    tail = &v13_.pNext;
  }
#endif
#ifdef VK_VERSION_1_4
  if (api_version >= VK_API_VERSION_1_4) {
    *tail = &v14_;
    tail = &v14_.pNext;
  }
#endif
  // Query the core features first: acceleration structures require bufferDeviceAddress.
  vkGetPhysicalDeviceFeatures2(device, &features_);
  if (v12_.bufferDeviceAddress && has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
      has(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
    *tail = &acceleration_;
    vkGetPhysicalDeviceFeatures2(device, &features_);
    if (acceleration_.accelerationStructure) {
      extensions_.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
      extensions_.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
      tail = &acceleration_.pNext;
      if (has(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
        extensions_.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        *tail = &ray_pipeline_;
        tail = &ray_pipeline_.pNext;
      }
      if (has(VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
        extensions_.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        *tail = &ray_query_;
        tail = &ray_query_.pNext;
      }
    } else {
      *tail = nullptr;
    }
  }
  // Mandatory when advertised by a portability implementation.
  if (has("VK_KHR_portability_subset"))
    extensions_.push_back("VK_KHR_portability_subset");
  // Enable every supported bit in the negotiated core and RHI extension feature chains.
  // No selection requirement is involved in this query or in vkCreateDevice.
  vkGetPhysicalDeviceFeatures2(device, &features_);
  valid_ = true;
}

}  // namespace grassland::rhi::backend::vulkan
