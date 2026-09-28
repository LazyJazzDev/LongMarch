#pragma once
#include "grassland/graphics/backend/vulkan/helper/instance.h"

namespace grassland::graphics::backend::vulkan {
VkPhysicalDeviceFeatures GetPhysicalDeviceFeatures(VkPhysicalDevice physical_device);
VkPhysicalDeviceProperties GetPhysicalDeviceProperties(VkPhysicalDevice physical_device);
VkPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physical_device);
uint64_t GetDeviceLocalMemorySize(VkPhysicalDevice physical_device);
std::vector<VkExtensionProperties> GetDeviceExtensions(VkPhysicalDevice physical_device);
std::vector<VkQueueFamilyProperties> GetQueueFamilyProperties(VkPhysicalDevice physical_device);
bool IsExtensionSupported(VkPhysicalDevice physical_device, const char *extension_name);
[[maybe_unused]] bool SupportGeometryShader(VkPhysicalDevice physical_device);
VkPhysicalDeviceRayTracingPipelinePropertiesKHR GetPhysicalDeviceRayTracingPipelineProperties(
    VkPhysicalDevice physical_device);
VkPhysicalDeviceRayTracingPipelineFeaturesKHR GetPhysicalDeviceRayTracingPipelineFeatures(
    VkPhysicalDevice physical_device);
bool SupportRayQuery(VkPhysicalDevice physical_device);
bool SupportRayTracing(VkPhysicalDevice physical_device);
uint64_t Evaluate(VkPhysicalDevice physical_device);
uint32_t GraphicsFamilyIndex(VkPhysicalDevice physical_device);
uint32_t PresentFamilyIndex(VkPhysicalDevice physical_device, VkSurfaceKHR surface);
uint32_t ComputeFamilyIndex(VkPhysicalDevice physical_device);
uint32_t TransferFamilyIndex(VkPhysicalDevice physical_device);
#if defined(LONGMARCH_CUDA_RUNTIME)
int GetCUDADeviceIndex(VkPhysicalDevice physical_device);
#endif
}  // namespace grassland::graphics::backend::vulkan
