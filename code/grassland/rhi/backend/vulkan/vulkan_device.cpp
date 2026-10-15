#include "grassland/rhi/backend/vulkan/vulkan_device.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

bool VulkanDevice::AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command, VkFence &fence) {
  {
    std::lock_guard<std::mutex> lock(resource_pool_mutex_);
    if (!command_pool_.empty()) {
      std::tie(pool, command, fence) = command_pool_.back();
      command_pool_.pop_back();
    }
  }
  if (pool) {
    if (!Check(vkResetCommandPool(device_, pool, 0), "Reset command pool"))
      return false;
    if (!Check(vkResetFences(device_, 1, &fence), "Reset submission fence"))
      return false;
  } else {
    VkCommandPoolCreateInfo create{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    create.queueFamilyIndex = family_;
    create.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (!Check(vkCreateCommandPool(device_, &create, nullptr, &pool), "Create command pool"))
      return false;
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Check(vkCreateFence(device_, &fence_info, nullptr, &fence), "Create submission fence"))
      return false;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (!Check(vkAllocateCommandBuffers(device_, &allocate, &command), "Allocate command buffer"))
      return false;
  }
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  return Check(vkBeginCommandBuffer(command, &begin), "Begin command buffer");
}

void VulkanDevice::RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command, VkFence fence) {
  try {
    std::lock_guard<std::mutex> lock(resource_pool_mutex_);
    command_pool_.emplace_back(pool, command, fence);
    return;
  } catch (const std::bad_alloc &) {
  }
  vkDestroyFence(device_, fence, nullptr);
  vkDestroyCommandPool(device_, pool, nullptr);
}

bool VulkanDevice::AcquireImageSync(VkSemaphore &semaphore, VkFence &fence) {
  {
    std::lock_guard<std::mutex> lock(resource_pool_mutex_);
    if (!image_sync_pool_.empty()) {
      std::tie(semaphore, fence) = image_sync_pool_.back();
      image_sync_pool_.pop_back();
    }
  }
  if (semaphore)
    return Check(vkResetFences(device_, 1, &fence), "Reset acquire fence");
  VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  if (!Check(vkCreateSemaphore(device_, &create, nullptr, &semaphore), "Create acquire semaphore"))
    return false;
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  return Check(vkCreateFence(device_, &fence_info, nullptr, &fence), "Create acquire fence");
}

void VulkanDevice::RecycleImageSync(VkSemaphore semaphore, VkFence fence) {
  try {
    std::lock_guard<std::mutex> lock(resource_pool_mutex_);
    image_sync_pool_.emplace_back(semaphore, fence);
    return;
  } catch (const std::bad_alloc &) {
  }
  vkDestroySemaphore(device_, semaphore, nullptr);
  vkDestroyFence(device_, fence, nullptr);
}

VulkanDevice::VulkanDevice(grassland::rhi::PhysicalDevice info, Core *core) : Device(std::move(info), core) {
}

VulkanDevice::~VulkanDevice() {
  ShutdownQueues();
  for (const auto &entry : command_pool_) {
    vkDestroyFence(device_, std::get<2>(entry), nullptr);
    vkDestroyCommandPool(device_, std::get<0>(entry), nullptr);
  }
  for (const auto &entry : image_sync_pool_) {
    vkDestroySemaphore(device_, entry.first, nullptr);
    vkDestroyFence(device_, entry.second, nullptr);
  }
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

rhi::IWindow *VulkanDevice::CreateGLFWWindow(const WindowDesc &desc) {
  try {
    auto window = std::make_unique<VulkanWindow>(this);
    if (!window->Initialize(desc))
      return nullptr;
    return window.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Window creation ran out of memory.");
    return nullptr;
  }
}

rhi::ICommandContext *VulkanDevice::CreateCommandContext() {
  try {
    auto context = std::make_unique<VulkanCommandContext>(this);
    if (!context->Initialize())
      return nullptr;
    return context.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Command context allocation failed.");
    return nullptr;
  }
}

rhi::IQueue *VulkanDevice::GetQueue(QueueType type) {
  try {
    return new VulkanQueue(this, type);
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Queue creation ran out of memory.");
    return nullptr;
  }
}

}  // namespace grassland::rhi::backend::vulkan
