#include "grassland/rhi/backend/vulkan/vulkan_device.h"

#include <vk_mem_alloc.h>

#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

bool VulkanDevice::AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command) {
  {
    std::lock_guard<std::mutex> lock(resource_pool_mutex_);
    if (!command_pool_.empty()) {
      std::tie(pool, command) = command_pool_.back();
      command_pool_.pop_back();
    }
  }
  if (pool) {
    if (!Check(vkResetCommandPool(device_, pool, 0), "Reset command pool"))
      return false;
  } else {
    VkCommandPoolCreateInfo create{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    create.queueFamilyIndex = graphics_queue_family_index_;
    create.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (!Check(vkCreateCommandPool(device_, &create, nullptr, &pool), "Create command pool"))
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

void VulkanDevice::RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command) {
  try {
    std::lock_guard<std::mutex> lock(resource_pool_mutex_);
    command_pool_.emplace_back(pool, command);
    return;
  } catch (const std::bad_alloc &) {
  }
  vkDestroyCommandPool(device_, pool, nullptr);
}

VulkanDevice::VulkanDevice(grassland::rhi::PhysicalDevice info, Core *core)
    : Device(std::move(info), core),
      upload_pool_(this) {
}

VulkanDevice::~VulkanDevice() {
  ShutdownQueues();
  upload_pool_.Clear();
  render_targets_.reset();
  for (const auto &entry : command_pool_) {
    vkDestroyCommandPool(device_, std::get<0>(entry), nullptr);
  }
  if (timeline_)
    vkDestroySemaphore(device_, timeline_, nullptr);
  if (allocator_)
    vmaDestroyAllocator(allocator_);
  if (device_)
    vkDestroyDevice(device_, nullptr);
}

VulkanDevice *VulkanDevice::CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                               Core *core,
                                               VkPhysicalDevice physical_device,
                                               uint32_t api_version) {
  auto device = std::unique_ptr<VulkanDevice>(new VulkanDevice(std::move(info), core));
  DeviceFeatures enabled(physical_device, api_version);
  if (!enabled)
    return {};
  if (!enabled.TimelineSemaphore()) {
    ReportError(-1, "Vulkan RHI requires timelineSemaphore support.");
    return {};
  }
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
  if (enabled.DynamicRendering()) {
    const char *begin_name = api_version >= VK_API_VERSION_1_3 ? "vkCmdBeginRendering" : "vkCmdBeginRenderingKHR";
    const char *end_name = api_version >= VK_API_VERSION_1_3 ? "vkCmdEndRendering" : "vkCmdEndRenderingKHR";
    device->begin_rendering_ =
        reinterpret_cast<PFN_vkCmdBeginRendering>(vkGetDeviceProcAddr(device->device_, begin_name));
    device->end_rendering_ = reinterpret_cast<PFN_vkCmdEndRendering>(vkGetDeviceProcAddr(device->device_, end_name));
    if (!device->begin_rendering_ || !device->end_rendering_) {
      ReportError(-1, "Enabled dynamic rendering commands are unavailable.");
      return {};
    }
  }
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphore.pNext = &type;
  if (!Check(vkCreateSemaphore(device->device_, &semaphore, nullptr, &device->timeline_), "Create submission timeline"))
    return {};
  const bool swapchain = std::any_of(enabled.Extensions().begin(), enabled.Extensions().end(), [](const char *name) {
    return std::strcmp(name, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
  });
  VmaAllocatorCreateInfo memory{};
  memory.physicalDevice = physical_device;
  memory.device = device->device_;
  memory.instance = static_cast<VulkanCore *>(core)->Instance();
  memory.vulkanApiVersion = VK_API_VERSION_1_2;
  if (!Check(vmaCreateAllocator(&memory, &device->allocator_), "Create VMA allocator"))
    return {};
  device->render_targets_ =
      std::make_unique<VulkanRenderTargetPool>(device->device_, device->begin_rendering_ != nullptr);
  device->physical_ = physical_device;
  device->graphics_queue_family_index_ = graphics_family;
  device->swapchain_enabled_ = swapchain;
  vkGetDeviceQueue(device->device_, graphics_family, 0, &device->queue_);
  return device.release();
}

VkSemaphore VulkanDevice::Timeline() const {
  return timeline_;
}

uint64_t VulkanDevice::NextSerial() {
  return ++next_serial_;
}

VkResult VulkanDevice::Wait(uint64_t value, uint64_t timeout) const {
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &timeline_;
  wait.pValues = &value;
  return vkWaitSemaphores(device_, &wait, timeout);
}

PFN_vkCmdBeginRendering VulkanDevice::BeginRenderingCommand() const {
  return begin_rendering_;
}

PFN_vkCmdEndRendering VulkanDevice::EndRenderingCommand() const {
  return end_rendering_;
}

VulkanRenderTargetPool &VulkanDevice::RenderTargets() {
  return *render_targets_;
}

VmaAllocator VulkanDevice::Allocator() const {
  return allocator_;
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
  return graphics_queue_family_index_;
}

bool VulkanDevice::SupportsSwapchain() const {
  return swapchain_enabled_;
}

std::mutex &VulkanDevice::QueueMutex() {
  return mutex_;
}

VulkanUploadPool &VulkanDevice::UploadPool() {
  return upload_pool_;
}

rhi::IImage *VulkanDevice::CreateImage(const ImageDesc &desc) {
  return VulkanAllocatedImage::Create(this, desc);
}

rhi::IBuffer *VulkanDevice::CreateBuffer(const BufferDesc &desc) {
  return VulkanAllocatedBuffer::Create(this, desc);
}

rhi::IProgram *VulkanDevice::CreateProgram(const ProgramDesc &desc) {
  try {
    return new VulkanProgram(this, desc);
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Program allocation failed.");
    return nullptr;
  }
}

rhi::IWindow *VulkanDevice::CreateGLFWWindow(const WindowDesc &desc) {
  return VulkanWindow::Create(this, desc);
}

rhi::ICommandContext *VulkanDevice::CreateCommandContext() {
  return VulkanCommandContext::Create(this);
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
