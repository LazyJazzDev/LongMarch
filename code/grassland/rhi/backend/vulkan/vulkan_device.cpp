#include "grassland/rhi/backend/vulkan/vulkan_device.h"

#include <vk_mem_alloc.h>

#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_ray_tracing_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"
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
    : Device(std::move(info), core, BACKEND_API_VULKAN),
      upload_pool_(this, TEMPORAL_BUFFER_UPLOAD),
      download_pool_(this, TEMPORAL_BUFFER_DOWNLOAD),
      scratch_pool_(this) {
}

VulkanDevice::~VulkanDevice() {
  ShutdownQueues();
  upload_pool_.Clear();
  download_pool_.Clear();
  scratch_pool_.Clear();
  render_targets_.reset();
  descriptor_pool_.reset();
  for (const auto &entry : command_pool_) {
    vkDestroyCommandPool(device_, std::get<0>(entry), nullptr);
  }
  if (allocator_)
    vmaDestroyAllocator(allocator_);
  ReleaseQueues();
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
  if (!enabled.ScalarBlockLayout()) {
    ReportError(-1, "Vulkan RHI requires scalarBlockLayout for DX-compatible buffer layouts.");
    return {};
  }
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
  std::vector<VkDeviceQueueCreateInfo> queues;
  uint32_t graphics_family = UINT32_MAX;
  const float priorities[] = {1.0f, 1.0f};
  for (uint32_t i = 0; i < count; ++i) {
    if (!families[i].queueCount)
      continue;
    if (graphics_family == UINT32_MAX && ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                                          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
      graphics_family = i;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = i;
    queue.queueCount = std::min(2u, families[i].queueCount);
    queue.pQueuePriorities = priorities;
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
  if (!device->procedures_.Load(device->device_, api_version, enabled))
    return {};
  const bool swapchain = std::any_of(enabled.Extensions().begin(), enabled.Extensions().end(), [](const char *name) {
    return std::strcmp(name, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
  });
  VmaAllocatorCreateInfo memory{};
  if (enabled.RayTracingPipeline() || enabled.RayQuery())
    memory.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  memory.physicalDevice = physical_device;
  memory.device = device->device_;
  memory.instance = static_cast<VulkanCore *>(core)->Instance();
  memory.vulkanApiVersion = VK_API_VERSION_1_2;
  if (!Check(vmaCreateAllocator(&memory, &device->allocator_), "Create VMA allocator"))
    return {};
  device->descriptor_pool_ = std::make_unique<VulkanDescriptorPool>(device->device_);
  device->render_targets_ =
      std::make_unique<VulkanRenderTargetPool>(device->device_, device->procedures_.CmdBeginRendering != nullptr);
  device->physical_ = physical_device;
  device->graphics_queue_family_index_ = graphics_family;
  device->swapchain_enabled_ = swapchain;
  VkQueue graphics{}, transfer{};
  vkGetDeviceQueue(device->device_, graphics_family, 0, &graphics);
  vkGetDeviceQueue(device->device_, graphics_family, families[graphics_family].queueCount > 1 ? 1 : 0, &transfer);
  auto graphics_queue = ObjectPtr<VulkanQueue>::Adopt(VulkanQueue::Create(device.get(), QUEUE_TYPE_GRAPHICS, graphics));
  if (!graphics_queue)
    return {};
  auto physical_owner = graphics == transfer ? graphics_queue.Get() : nullptr;
  device->InitializeQueue(std::move(graphics_queue));
  auto transfer_queue =
      ObjectPtr<VulkanQueue>::Adopt(VulkanQueue::Create(device.get(), QUEUE_TYPE_TRANSFER, transfer, physical_owner));
  if (!transfer_queue)
    return {};
  device->InitializeQueue(std::move(transfer_queue));
  return device.release();
}

VkSemaphore VulkanDevice::Timeline(QueueType type) const {
  return static_cast<VulkanQueue *>(QueueState(type))->Timeline();
}

uint64_t VulkanDevice::NextSerial(QueueType type) {
  return QueueState(type)->NextSerial();
}

VkResult VulkanDevice::Wait(uint64_t value, uint64_t timeout, QueueType type) const {
  return static_cast<VulkanQueue *>(QueueState(type))->Wait(value, timeout);
}

const DeviceProcedures &VulkanDevice::Procedures() const {
  return procedures_;
}

ISampler *VulkanDevice::CreateSampler(const SamplerDesc &desc) {
  return VulkanSampler::Create(this, desc);
}

IBindingGroup *VulkanDevice::CreateBindingGroup(const BindingLayoutDesc &layout) {
  return VulkanBindingGroup::Create(this, layout);
}

VulkanDescriptorPool &VulkanDevice::DescriptorPool() {
  return *descriptor_pool_;
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

VkQueue VulkanDevice::NativeQueue(QueueType type) const {
  return static_cast<VulkanQueue *>(QueueState(type))->Native();
}

uint32_t VulkanDevice::QueueFamily() const {
  return graphics_queue_family_index_;
}

bool VulkanDevice::SupportsSwapchain() const {
  return swapchain_enabled_;
}

std::mutex &VulkanDevice::QueueMutex(QueueType type) {
  return QueueState(type)->Mutex();
}

VulkanTemporalBufferPool &VulkanDevice::UploadPool() {
  return upload_pool_;
}

rhi::IImage *VulkanDevice::CreateImage(const ImageDesc &desc) {
  return VulkanAllocatedImage::Create(this, desc);
}

rhi::IBuffer *VulkanDevice::CreateBuffer(const BufferDesc &desc) {
  return VulkanAllocatedBuffer::Create(this, desc);
}

rhi::IComputeProgram *VulkanDevice::CreateComputeProgram(const ComputeProgramDesc &desc) {
  return VulkanComputeProgram::Create(this, desc);
}

rhi::IProgram *VulkanDevice::CreateProgram(const ProgramDesc &desc) {
  return VulkanProgram::Create(this, desc);
}

rhi::IWindow *VulkanDevice::CreateGLFWWindow(const WindowDesc &desc) {
  return VulkanWindow::Create(this, desc);
}

}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
VulkanScratchPool &VulkanDevice::ScratchPool() {
  return scratch_pool_;
}
}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
IBottomLevelAccelerationStructure *VulkanDevice::CreateBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructureDesc &desc) {
  return VulkanBottomLevelAccelerationStructure::Create(this, desc);
}

ITopLevelAccelerationStructure *VulkanDevice::CreateTopLevelAccelerationStructure(
    const TopLevelAccelerationStructureDesc &desc) {
  return VulkanTopLevelAccelerationStructure::Create(this, desc);
}

IRayTracingProgram *VulkanDevice::CreateRayTracingProgram(const RayTracingProgramDesc &desc) {
  if (!SupportRayTracingPipeline())
    return nullptr;
  return VulkanRayTracingProgram::Create(this, desc);
}

VulkanTemporalBufferPool &VulkanDevice::DownloadPool() {
  return download_pool_;
}
}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {

bool VulkanDevice::WaitTimelines(const std::vector<Event> &points, bool wait_all) const {
  // Coalesce duplicate timeline objects: max for All, min for Any.
  const auto unique = CoalesceEvents(points, wait_all);
  std::vector<VkSemaphore> semaphores;
  std::vector<uint64_t> values;
  for (auto event : unique) {
    semaphores.push_back(static_cast<VulkanTimeline *>(event.timeline)->Native());
    values.push_back(event.value);
  }
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.flags = wait_all ? 0 : VK_SEMAPHORE_WAIT_ANY_BIT;
  wait.semaphoreCount = static_cast<uint32_t>(semaphores.size());
  wait.pSemaphores = semaphores.data();
  wait.pValues = values.data();
  return vkWaitSemaphores(device_, &wait, UINT64_MAX) == VK_SUCCESS;
}

void VulkanDevice::WakeCollector() {
  std::vector<Event> wake;
  for (auto queue : {QUEUE_TYPE_GRAPHICS, QUEUE_TYPE_TRANSFER}) {
    if (!NativeQueue(queue) || !Timeline(queue))
      continue;
    std::lock_guard<std::mutex> lock(QueueMutex(queue));
    auto value = NextSerial(queue);
    auto semaphore = Timeline(queue);
    VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues = &value;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &semaphore;
    if (Check(vkQueueSubmit(NativeQueue(queue), 1, &submit, VK_NULL_HANDLE), "Signal collector shutdown"))
      wake.push_back({QueueState(queue)->TimelineState(), value});
  }
  if (!wake.empty())
    WaitTimelines(wake, true);
}

}  // namespace grassland::rhi::backend::vulkan
