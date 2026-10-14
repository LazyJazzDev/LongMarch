#include "grassland/rhi/backend/vulkan/vulkan_device.h"

#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanDevice::VulkanDevice(PhysicalDevice info, const ObjectPtr<ICore> &core) : IDevice(std::move(info), core) {
}

VulkanDevice::~VulkanDevice() {
  ShutdownQueues();
  // device_ is non-null only until ownership transfers to BackendLifetime.
  if (device_)
    vkDestroyDevice(device_, nullptr);
}

std::unique_ptr<VulkanDevice> VulkanDevice::CreateVulkanDevice(PhysicalDevice info,
                                                               const ObjectPtr<ICore> &core,
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
  device->lifetime_ =
      std::make_shared<BackendLifetime>(device->device_, physical_device, core, graphics_family, swapchain);
  device->device_ = VK_NULL_HANDLE;
  return device;
}

VulkanDevice::BackendLifetime::BackendLifetime(VkDevice device,
                                               VkPhysicalDevice physical,
                                               const ObjectPtr<ICore> &core,
                                               uint32_t family,
                                               bool swapchain)
    : core_(core),
      device_(device),
      physical_(physical),
      family_(family),
      swapchain_(swapchain) {
  vkGetDeviceQueue(device_, family_, 0, &queue_);
}

VulkanDevice::BackendLifetime::~BackendLifetime() {
  vkDestroyDevice(device_, nullptr);
}

VkDevice VulkanDevice::BackendLifetime::Device() const {
  return device_;
}

VkPhysicalDevice VulkanDevice::BackendLifetime::PhysicalDevice() const {
  return physical_;
}

VkInstance VulkanDevice::BackendLifetime::Instance() const {
  return static_cast<VulkanCore *>(core_.Get())->Instance();
}

VkQueue VulkanDevice::BackendLifetime::Queue() const {
  return queue_;
}

uint32_t VulkanDevice::BackendLifetime::QueueFamily() const {
  return family_;
}

bool VulkanDevice::BackendLifetime::SupportsSwapchain() const {
  return swapchain_;
}

std::mutex &VulkanDevice::BackendLifetime::QueueMutex() {
  return mutex_;
}

const std::shared_ptr<VulkanDevice::BackendLifetime> &VulkanDevice::NativeLifetime() const {
  return lifetime_;
}

const void *VulkanDevice::NativeIdentity() const {
  return lifetime_.get();
}

std::unique_ptr<IWindow> VulkanDevice::CreateWindow(const WindowDesc &desc) {
  auto window = std::make_unique<VulkanWindow>(lifetime_);
  return window->Initialize(desc) ? std::unique_ptr<IWindow>(std::move(window)) : nullptr;
}

std::unique_ptr<IQueue> VulkanDevice::CreateQueue(QueueType type) {
  return std::make_unique<VulkanQueue>(ObjectPtr<VulkanDevice>::Retain(this), type);
}

}  // namespace grassland::rhi::backend::vulkan
