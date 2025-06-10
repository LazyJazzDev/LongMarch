#include "grassland/vulkan/physical_device.h"

#include "grassland/vulkan/device.h"
#include "grassland/vulkan/surface.h"

namespace grassland::vulkan {
PhysicalDevice::PhysicalDevice(VkPhysicalDevice physical_device) {
  physical_device_ = physical_device;
}

PhysicalDevice::~PhysicalDevice() = default;

VkPhysicalDevice PhysicalDevice::Handle() const {
  return physical_device_;
}

VkPhysicalDeviceFeatures PhysicalDevice::GetPhysicalDeviceFeatures() const {
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(physical_device_, &features);
  return features;
}

VkPhysicalDeviceProperties PhysicalDevice::GetPhysicalDeviceProperties() const {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device_, &properties);
  return properties;
}

VkPhysicalDeviceMemoryProperties PhysicalDevice::GetPhysicalDeviceMemoryProperties() const {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &properties);
  return properties;
}

uint64_t PhysicalDevice::GetDeviceLocalMemorySize() const {
  VkPhysicalDeviceMemoryProperties properties = GetPhysicalDeviceMemoryProperties();
  uint64_t device_local_memory_size = 0;
  for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
    if (properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      device_local_memory_size = properties.memoryHeaps[i].size;
    }
  }
  return device_local_memory_size;
}

std::vector<VkExtensionProperties> PhysicalDevice::GetDeviceExtensions() const {
  uint32_t extension_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, extensions.data());
  return extensions;
}

std::vector<VkQueueFamilyProperties> PhysicalDevice::GetQueueFamilyProperties() const {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());
  return queue_families;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR PhysicalDevice::GetRayTracingProperties() const {
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties{};
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &ray_tracing_properties;
  ray_tracing_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
  return ray_tracing_properties;
}

bool PhysicalDevice::IsExtensionSupported(const char *extension_name) const {
  std::vector<VkExtensionProperties> extensions = GetDeviceExtensions();
  for (const auto &extension : extensions) {
    if (strcmp(extension.extensionName, extension_name) == 0) {
      return true;
    }
  }
  return false;
}

[[maybe_unused]] bool PhysicalDevice::SupportGeometryShader() const {
  // Geometry shader is feature of Vulkan
  VkPhysicalDeviceFeatures features = GetPhysicalDeviceFeatures();
  return features.geometryShader;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR PhysicalDevice::GetPhysicalDeviceRayTracingPipelineProperties() const {
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  properties.pNext = nullptr;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &properties;
  vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
  return properties;
}

VkPhysicalDeviceRayTracingPipelineFeaturesKHR PhysicalDevice::GetPhysicalDeviceRayTracingPipelineFeatures() const {
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  features.pNext = nullptr;
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &features;
  vkGetPhysicalDeviceFeatures2(physical_device_, &features2);
  return features;
}

bool PhysicalDevice::SupportRayTracing() const {
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR features = GetPhysicalDeviceRayTracingPipelineFeatures();
  return features.rayTracingPipeline;
}

uint64_t PhysicalDevice::Evaluate() const {
  uint64_t score = 0;
  VkPhysicalDeviceProperties properties = GetPhysicalDeviceProperties();
  VkPhysicalDeviceFeatures features = GetPhysicalDeviceFeatures();
  VkPhysicalDeviceMemoryProperties memory_properties = GetPhysicalDeviceMemoryProperties();
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties =
      GetPhysicalDeviceRayTracingPipelineProperties();
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_features = GetPhysicalDeviceRayTracingPipelineFeatures();
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
  uint64_t device_local_memory_size = GetDeviceLocalMemorySize();

  // Score for memory size
  score += device_local_memory_size / 1000000;

  return score;
}

uint32_t PhysicalDevice::GraphicsFamilyIndex() const {
  uint32_t graphics_family_index = 0;
  std::vector<VkQueueFamilyProperties> queue_families = GetQueueFamilyProperties();
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      return graphics_family_index;
    }
    graphics_family_index++;
  }
  return -1;
}

uint32_t PhysicalDevice::PresentFamilyIndex(const Surface *surface) const {
  uint32_t present_family_index = 0;
  std::vector<VkQueueFamilyProperties> queue_families = GetQueueFamilyProperties();
  for (const auto &queue_family : queue_families) {
    VkBool32 present_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, present_family_index, surface->Handle(), &present_support);
    if (present_support) {
      return present_family_index;
    }
    present_family_index++;
  }
  return -1;
}

uint32_t PhysicalDevice::ComputeFamilyIndex() const {
  uint32_t compute_family_index = 0;
  std::vector<VkQueueFamilyProperties> queue_families = GetQueueFamilyProperties();
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      return compute_family_index;
    }
    compute_family_index++;
  }
  return -1;
}

uint32_t PhysicalDevice::TransferFamilyIndex() const {
  uint32_t transfer_family_index = 0;
  std::vector<VkQueueFamilyProperties> queue_families = GetQueueFamilyProperties();
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT) {
      return transfer_family_index;
    }
    transfer_family_index++;
  }
  return -1;
}

VkSampleCountFlagBits PhysicalDevice::GetMaxUsableSampleCount() const {
  VkPhysicalDeviceProperties physical_device_properties = GetPhysicalDeviceProperties();

  VkSampleCountFlags counts = physical_device_properties.limits.framebufferColorSampleCounts &
                              physical_device_properties.limits.framebufferDepthSampleCounts;
  if (counts & VK_SAMPLE_COUNT_64_BIT) {
    return VK_SAMPLE_COUNT_64_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_32_BIT) {
    return VK_SAMPLE_COUNT_32_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_16_BIT) {
    return VK_SAMPLE_COUNT_16_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_8_BIT) {
    return VK_SAMPLE_COUNT_8_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_4_BIT) {
    return VK_SAMPLE_COUNT_4_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_2_BIT) {
    return VK_SAMPLE_COUNT_2_BIT;
  }
  return VK_SAMPLE_COUNT_1_BIT;
}

bool PhysicalDevice::CheckFeatureSupport(const DeviceFeatureRequirement &feature_requirement) const {
  if (GraphicsFamilyIndex() == -1) {
    return false;
  }

  if (ComputeFamilyIndex() == -1) {
    return false;
  }

  if (TransferFamilyIndex() == -1) {
    return false;
  }

  if (feature_requirement.enable_raytracing_extension) {
    if (!SupportRayTracing()) {
      return false;
    }
  }

  return true;
}

#if defined(LONGMARCH_CUDA_RUNTIME)
int PhysicalDevice::GetCUDADeviceIndex() const {
  VkPhysicalDeviceIDProperties id_properties{};
  id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  id_properties.pNext = nullptr;

  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &id_properties;
  vkGetPhysicalDeviceProperties2(physical_device_, &properties2);

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

}  // namespace grassland::vulkan
