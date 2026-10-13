#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

VulkanDevice::VulkanDevice(PhysicalDevice info, const ObjectRef &core) : IDevice(std::move(info), core) {
}

VulkanDevice::~VulkanDevice() {
  // No submission API exists yet, so this device cannot have work in flight.
  if (device_)
    vkDestroyDevice(device_, nullptr);
}

std::unique_ptr<VulkanDevice> VulkanDevice::CreateVulkanDevice(PhysicalDevice info,
                                                               const ObjectRef &core,
                                                               VkPhysicalDevice physical_device,
                                                               uint32_t api_version) {
  auto device = std::unique_ptr<VulkanDevice>(new VulkanDevice(std::move(info), core));
  DeviceFeatures enabled(physical_device, api_version);
  if (!enabled)
    return {};
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
  std::vector<VkDeviceQueueCreateInfo> queues;
  const float priority = 1.0f;
  for (uint32_t i = 0; i < count; ++i) {
    if (!families[i].queueCount)
      continue;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = i;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    queues.push_back(queue);
  }
  VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  create.pNext = &enabled.Features();
  create.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
  create.pQueueCreateInfos = queues.data();
  create.enabledExtensionCount = static_cast<uint32_t>(enabled.Extensions().size());
  create.ppEnabledExtensionNames = enabled.Extensions().data();
  const VkResult result = vkCreateDevice(physical_device, &create, nullptr, &device->device_);
  if (result != VK_SUCCESS) {
    ReportError(result, "vkCreateDevice failed.");
    return {};
  }
  return device;
}

}  // namespace grassland::rhi::backend::vulkan
