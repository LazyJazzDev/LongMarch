#include "grassland/graphics/backend/vulkan/helper/physical_device.h"

namespace grassland::graphics::backend::vulkan {
VkPhysicalDeviceFeatures GetPhysicalDeviceFeatures(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(physical_device, &features);
  return features;
}

VkPhysicalDeviceProperties GetPhysicalDeviceProperties(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  return properties;
}

VkPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
  return properties;
}

uint64_t GetDeviceLocalMemorySize(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceMemoryProperties properties = GetPhysicalDeviceMemoryProperties(physical_device);
  uint64_t device_local_memory_size = 0;
  for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
    if (properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      device_local_memory_size = properties.memoryHeaps[i].size;
    }
  }
  return device_local_memory_size;
}

std::vector<VkExtensionProperties> GetDeviceExtensions(VkPhysicalDevice physical_device) {
  uint32_t extension_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data());
  return extensions;
}

std::vector<VkQueueFamilyProperties> GetQueueFamilyProperties(VkPhysicalDevice physical_device) {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
  return queue_families;
}

bool IsExtensionSupported(VkPhysicalDevice physical_device, const char *extension_name) {
  std::vector<VkExtensionProperties> extensions = GetDeviceExtensions(physical_device);
  for (const auto &extension : extensions) {
    if (strcmp(extension.extensionName, extension_name) == 0) {
      return true;
    }
  }
  return false;
}

[[maybe_unused]] bool SupportGeometryShader(VkPhysicalDevice physical_device) {
  // Geometry shader is feature of Vulkan
  VkPhysicalDeviceFeatures features = GetPhysicalDeviceFeatures(physical_device);
  return features.geometryShader;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR GetPhysicalDeviceRayTracingPipelineProperties(
    VkPhysicalDevice physical_device) {
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  properties.pNext = nullptr;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &properties;
  vkGetPhysicalDeviceProperties2(physical_device, &properties2);
  return properties;
}

VkPhysicalDeviceRayTracingPipelineFeaturesKHR GetPhysicalDeviceRayTracingPipelineFeatures(
    VkPhysicalDevice physical_device) {
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  features.pNext = nullptr;
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &features;
  vkGetPhysicalDeviceFeatures2(physical_device, &features2);
  return features;
}

bool SupportRayQuery(VkPhysicalDevice physical_device) {
  if (!IsExtensionSupported(physical_device, VK_KHR_RAY_QUERY_EXTENSION_NAME) ||
      !IsExtensionSupported(physical_device, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) ||
      !IsExtensionSupported(physical_device, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) ||
      !IsExtensionSupported(physical_device, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
    return false;
  VkPhysicalDeviceBufferDeviceAddressFeatures address{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &query;
  query.pNext = &acceleration;
  acceleration.pNext = &address;
  vkGetPhysicalDeviceFeatures2(physical_device, &features);
  return query.rayQuery && acceleration.accelerationStructure && address.bufferDeviceAddress;
}

bool SupportRayTracing(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR features = GetPhysicalDeviceRayTracingPipelineFeatures(physical_device);
  return features.rayTracingPipeline;
}

uint64_t Evaluate(VkPhysicalDevice physical_device) {
  uint64_t score = 0;
  VkPhysicalDeviceProperties properties = GetPhysicalDeviceProperties(physical_device);
  VkPhysicalDeviceFeatures features = GetPhysicalDeviceFeatures(physical_device);
  VkPhysicalDeviceMemoryProperties memory_properties = GetPhysicalDeviceMemoryProperties(physical_device);
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties =
      GetPhysicalDeviceRayTracingPipelineProperties(physical_device);
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_features =
      GetPhysicalDeviceRayTracingPipelineFeatures(physical_device);
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 1000000000;
  }
  // Design a score system, key features: geometry and raytracing
  if (features.geometryShader) {
    score += 1000000;
  }
  if (ray_tracing_features.rayTracingPipeline) {
    score += 100000000;
  }
  // Consider memory size, get from existed function
  uint64_t device_local_memory_size = GetDeviceLocalMemorySize(physical_device);

  // Score for memory size
  score += device_local_memory_size / 1000000;

  return score;
}

uint32_t GraphicsFamilyIndex(VkPhysicalDevice physical_device) {
  uint32_t graphics_family_index = 0;
  std::vector<VkQueueFamilyProperties> queue_families = GetQueueFamilyProperties(physical_device);
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      return graphics_family_index;
    }
    graphics_family_index++;
  }
  return -1;
}

uint32_t PresentFamilyIndex(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
  uint32_t present_family_index = 0;
  std::vector<VkQueueFamilyProperties> queue_families = GetQueueFamilyProperties(physical_device);
  for (const auto &queue_family : queue_families) {
    VkBool32 present_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, present_family_index, surface, &present_support);
    if (present_support) {
      return present_family_index;
    }
    present_family_index++;
  }
  return -1;
}

uint32_t ComputeFamilyIndex(VkPhysicalDevice physical_device) {
  uint32_t compute_family_index = 0;
  std::vector<VkQueueFamilyProperties> queue_families = GetQueueFamilyProperties(physical_device);
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      return compute_family_index;
    }
    compute_family_index++;
  }
  return -1;
}

uint32_t TransferFamilyIndex(VkPhysicalDevice physical_device) {
  uint32_t transfer_family_index = 0;
  std::vector<VkQueueFamilyProperties> queue_families = GetQueueFamilyProperties(physical_device);
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT) {
      return transfer_family_index;
    }
    transfer_family_index++;
  }
  return -1;
}

#if defined(LONGMARCH_CUDA_RUNTIME)
int GetCUDADeviceIndex(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceIDProperties id_properties{};
  id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  id_properties.pNext = nullptr;

  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &id_properties;
  vkGetPhysicalDeviceProperties2(physical_device, &properties2);

  int cuda_device_count = 0;
  cudaGetDeviceCount(&cuda_device_count);
  for (int i = 0; i < cuda_device_count; i++) {
    cudaDeviceProp device_properties{};
    cudaGetDeviceProperties(&device_properties, i);
    if (std::memcmp(id_properties.deviceUUID, &device_properties.uuid, VK_UUID_SIZE) == 0) {
      return i;
    }
  }
  return -1;
}

#endif
}  // namespace grassland::graphics::backend::vulkan
