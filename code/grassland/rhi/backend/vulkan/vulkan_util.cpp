#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
bool SupportsImageStorage(VkPhysicalDevice device, ImageFormat format) {
  const auto native = NativeImageFormat(format);
  if (native == VK_FORMAT_UNDEFINED || IsDepthFormat(format))
    return false;
  VkFormatProperties properties{};
  vkGetPhysicalDeviceFormatProperties(device, native, &properties);
  return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

VkFormat NativeImageFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_RGBA8_UNORM:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case IMAGE_FORMAT_BGRA8_UNORM:
      return VK_FORMAT_B8G8R8A8_UNORM;
    case IMAGE_FORMAT_D32_FLOAT:
      return VK_FORMAT_D32_SFLOAT;
    case IMAGE_FORMAT_D24_UNORM_S8_UINT:
      return VK_FORMAT_D24_UNORM_S8_UINT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

VkImageAspectFlags ImageAspect(ImageFormat format) {
  return IsDepthFormat(format) ? VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
                               : VK_IMAGE_ASPECT_COLOR_BIT;
}

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
  if (has(VK_KHR_SWAPCHAIN_EXTENSION_NAME))
    extensions_.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
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
  // Vulkan 1.2 uses the extension feature instead of the promoted 1.3 structure.
  if (api_version < VK_API_VERSION_1_3 && has(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
    extensions_.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    *tail = &dynamic_rendering_;
    tail = &dynamic_rendering_.pNext;
  }
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

bool Check(VkResult result, const char *operation) {
  if (result == VK_SUCCESS)
    return true;
  ReportError(result, operation);
  return false;
}

DeviceFeatures::operator bool() const noexcept {
  return valid_;
}

const VkPhysicalDeviceFeatures2 &DeviceFeatures::Features() const noexcept {
  return features_;
}

const std::vector<const char *> &DeviceFeatures::Extensions() const noexcept {
  return extensions_;
}

bool DeviceFeatures::RayTracingPipeline() const {
  return ray_pipeline_.rayTracingPipeline == VK_TRUE;
}

bool DeviceFeatures::ScalarBlockLayout() const {
  return v12_.scalarBlockLayout == VK_TRUE;
}

bool DeviceFeatures::TimelineSemaphore() const {
  return v12_.timelineSemaphore == VK_TRUE;
}

bool DeviceFeatures::DynamicRendering() const {
#ifdef VK_VERSION_1_3
  if (v13_.dynamicRendering)
    return true;
#endif
  return dynamic_rendering_.dynamicRendering == VK_TRUE;
}

bool DeviceFeatures::RayQuery() const {
  return ray_query_.rayQuery == VK_TRUE;
}

}  // namespace grassland::rhi::backend::vulkan
