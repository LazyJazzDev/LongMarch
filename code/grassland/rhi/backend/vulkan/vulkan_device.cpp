#include "grassland/rhi/backend/vulkan/vulkan_device.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanDevice::VulkanDevice(grassland::rhi::PhysicalDevice info, Core *core) : Device(std::move(info), core) {
}

VulkanDevice::~VulkanDevice() {
  ShutdownQueues();
  if (device_)
    vkDestroyDevice(device_, nullptr);
}

std::unique_ptr<VulkanDevice> VulkanDevice::CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                                               Core *core,
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
  uint32_t graphics_family = UINT32_MAX;
  const float priority = 1.0f;
  for (uint32_t i = 0; i < count; ++i) {
    if (!families[i].queueCount)
      continue;
    if (graphics_family == UINT32_MAX && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      graphics_family = i;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = i;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    queues.push_back(queue);
  }
  VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  if (graphics_family == UINT32_MAX) {
    ReportError(-1, "No Vulkan graphics queue is available.");
    return {};
  }
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
  const bool swapchain = std::any_of(enabled.Extensions().begin(), enabled.Extensions().end(), [](const char *name) {
    return std::strcmp(name, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
  });
  device->physical_ = physical_device;
  device->family_ = graphics_family;
  device->swapchain_ = swapchain;
  vkGetDeviceQueue(device->device_, graphics_family, 0, &device->queue_);
  return device;
}

VkDevice VulkanDevice::NativeDevice() const {
  return device_;
}

VkPhysicalDevice VulkanDevice::NativePhysicalDevice() const {
  return physical_;
}

VkInstance VulkanDevice::Instance() const {
  return static_cast<VulkanCore *>(OwnerCore())->Instance();
}

VkQueue VulkanDevice::NativeQueue() const {
  return queue_;
}

uint32_t VulkanDevice::QueueFamily() const {
  return family_;
}

bool VulkanDevice::SupportsSwapchain() const {
  return swapchain_;
}

std::mutex &VulkanDevice::QueueMutex() {
  return mutex_;
}

std::unique_ptr<Window> VulkanDevice::CreateNativeWindow(const WindowDesc &desc) {
  auto window = std::make_unique<VulkanWindow>(this);
  return window->Initialize(desc) ? std::unique_ptr<Window>(std::move(window)) : nullptr;
}

std::unique_ptr<CommandContext> VulkanDevice::CreateNativeCommandContext() {
  auto context = std::make_unique<VulkanCommandContext>(this);
  if (!context->Initialize())
    return {};
  return context;
}

std::unique_ptr<Queue> VulkanDevice::CreateQueue(QueueType type) {
  return std::make_unique<VulkanQueue>(this, type);
}

}  // namespace grassland::rhi::backend::vulkan
