#include "grassland/rhi/backend/vulkan/vulkan_device.h"

#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <list>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_shader.h"
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_ray_tracing_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend::vulkan {
bool VulkanDeviceImpl::WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const {
  std::vector<VkSemaphore> semaphores;
  std::vector<uint64_t> values;
  for (const auto &event : events) {
    semaphores.push_back(rhi::QueryInterfacePtr<IVulkanTimeline>(event.timeline).Get()->NativeSemaphore());
    values.push_back(event.value);
  }
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.flags = wait_all ? 0 : VK_SEMAPHORE_WAIT_ANY_BIT;
  info.semaphoreCount = static_cast<uint32_t>(events.size());
  info.pSemaphores = semaphores.data();
  info.pValues = values.data();
  return Check(vkWaitSemaphores(device_, &info, UINT64_MAX), "Wait for timeline events");
}

void VulkanDeviceImpl::InterruptCollector() {
  VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
  signal.semaphore = collector_shutdown_;
  signal.value = 1;
  Check(vkSignalSemaphore(device_, &signal), "Stop device collector");
}

void VulkanDeviceImpl::WaitForProgress(const std::vector<Event> &next) {
  std::array<VkSemaphore, 3> semaphores{};
  std::array<uint64_t, 3> values{};
  for (size_t index = 0; index < next.size(); ++index) {
    semaphores[index] = rhi::QueryInterfacePtr<IVulkanTimeline>(next[index].timeline).Get()->NativeSemaphore();
    values[index] = next[index].value;
  }
  size_t count = next.size();
  if (!shutdown_consumed_) {
    semaphores[count] = collector_shutdown_;
    values[count++] = 1;
  }
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
  info.semaphoreCount = static_cast<uint32_t>(count);
  info.pSemaphores = semaphores.data();
  info.pValues = values.data();
  vkWaitSemaphores(device_, &info, UINT64_MAX);
  // Consume the one-shot shutdown notification, then drain real GPU work.
  shutdown_consumed_ = stopping_;
}

bool VulkanDeviceImpl::AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command) {
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
    create.queueFamilyIndex = queue_family_index_;
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

void VulkanDeviceImpl::RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command) {
  try {
    std::lock_guard<std::mutex> lock(resource_pool_mutex_);
    command_pool_.emplace_back(pool, command);
    return;
  } catch (const std::bad_alloc &) {
  }
  vkDestroyCommandPool(device_, pool, nullptr);
}

VulkanDeviceImpl::VulkanDeviceImpl(grassland::rhi::PhysicalDevice info, IVulkanCore *core)
    : upload_pool_(this, TEMPORAL_BUFFER_UPLOAD),
      download_pool_(this, TEMPORAL_BUFFER_DOWNLOAD),
      scratch_pool_(this) {
  InitializeDevice(std::move(info), core, BACKEND_API_VULKAN);
}

VulkanDeviceImpl::~VulkanDeviceImpl() {
  ShutdownQueues();
  upload_pool_.Clear();
  download_pool_.Clear();
  scratch_pool_.Clear();
  render_targets_.reset();
  descriptor_pool_.reset();
  for (const auto &entry : command_pool_) {
    vkDestroyCommandPool(device_, std::get<0>(entry), nullptr);
  }
  for (auto &queue : queues_)
    queue.Reset();
  if (collector_shutdown_)
    vkDestroySemaphore(device_, collector_shutdown_, nullptr);
  if (allocator_)
    vmaDestroyAllocator(allocator_);
  if (device_)
    vkDestroyDevice(device_, nullptr);

  DestroyDevice();
}

rhi::IDevice *VulkanDeviceImpl::CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                                   IVulkanCore *core,
                                                   VkPhysicalDevice physical_device,
                                                   uint32_t api_version) {
  auto device = std::unique_ptr<VulkanDeviceImpl>(new VulkanDeviceImpl(std::move(info), core));
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
  uint32_t queue_family = UINT32_MAX;
  for (uint32_t i = 0; i < count; ++i) {
    const auto required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if (!families[i].queueCount || (families[i].queueFlags & required) != required)
      continue;
    if (queue_family == UINT32_MAX)
      queue_family = i;
    if (families[i].queueCount >= 2) {
      queue_family = i;
      break;
    }
  }
  if (queue_family == UINT32_MAX) {
    ReportError(-1, "No Vulkan graphics and compute queue is available.");
    return {};
  }
  // Both roles share a family, so ordinary resources need no ownership transfers.
  // Do not create unused queues in unrelated families.
  const float priorities[] = {1.0f, 1.0f};
  VkDeviceQueueCreateInfo queue_create{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_create.queueFamilyIndex = queue_family;
  queue_create.queueCount = std::min(2u, families[queue_family].queueCount);
  queue_create.pQueuePriorities = priorities;
  VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  create.pNext = &enabled.Features();
  create.queueCreateInfoCount = 1;
  create.pQueueCreateInfos = &queue_create;
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
  memory.instance = core->Instance();
  memory.vulkanApiVersion = VK_API_VERSION_1_2;
  if (!Check(vmaCreateAllocator(&memory, &device->allocator_), "Create VMA allocator"))
    return {};
  device->descriptor_pool_ = std::make_unique<VulkanDescriptorPool>(device->device_);
  device->render_targets_ =
      std::make_unique<VulkanRenderTargetPool>(device->device_, device->procedures_.CmdBeginRendering != nullptr);
  device->physical_ = physical_device;
  device->queue_family_index_ = queue_family;
  device->swapchain_enabled_ = swapchain;
  auto main_mutex = std::make_shared<std::mutex>();
  for (uint32_t index = 0; index < 2; ++index) {
    const auto native_index = index < queue_create.queueCount ? index : 0u;
    VkQueue queue;
    vkGetDeviceQueue(device->device_, queue_family, native_index, &queue);
    auto mutex = native_index == 0 ? main_mutex : std::make_shared<std::mutex>();
    device->queues_[index] = ObjectPtr<IVulkanQueue>::Adopt(
        IVulkanQueue::Create(device.get(), static_cast<QueueType>(index), queue, std::move(mutex)));
    if (!device->queues_[index])
      return nullptr;
  }
  VkSemaphoreTypeCreateInfo shutdown_type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  shutdown_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo shutdown_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  shutdown_info.pNext = &shutdown_type;
  if (!Check(vkCreateSemaphore(device->device_, &shutdown_info, nullptr, &device->collector_shutdown_),
             "Create collector shutdown semaphore"))
    return nullptr;
  device->StartCollector();
  return device.release();
}

const DeviceProcedures &VulkanDeviceImpl::Procedures() const {
  return procedures_;
}

ISampler *VulkanDeviceImpl::CreateSampler(const SamplerDesc &desc) {
  return IVulkanSampler::Create(this, desc);
}

IBindingGroup *VulkanDeviceImpl::CreateBindingGroup(const BindingLayoutDesc &layout) {
  return CreateVulkanBindingGroup(this, layout);
}

VulkanDescriptorPool &VulkanDeviceImpl::DescriptorPool() {
  return *descriptor_pool_;
}

VulkanRenderTargetPool &VulkanDeviceImpl::RenderTargets() {
  return *render_targets_;
}

VmaAllocator VulkanDeviceImpl::Allocator() const {
  return allocator_;
}

VkDevice VulkanDeviceImpl::NativeDevice() const {
  return device_;
}

VkPhysicalDevice VulkanDeviceImpl::NativePhysicalDevice() const {
  return physical_;
}

VkInstance VulkanDeviceImpl::Instance() const {
  return core_->Instance();
}

uint32_t VulkanDeviceImpl::QueueFamily() const {
  return queue_family_index_;
}

bool VulkanDeviceImpl::SupportsSwapchain() const {
  return swapchain_enabled_;
}

VulkanTemporalBufferPool &VulkanDeviceImpl::UploadPool() {
  return upload_pool_;
}

rhi::IImage *VulkanDeviceImpl::CreateImage(const ImageDesc &desc) {
  return CreateVulkanAllocatedImage(this, desc);
}

rhi::IBuffer *VulkanDeviceImpl::CreateBuffer(const BufferDesc &desc) {
  auto buffer = ObjectPtr<IVulkanAllocatedBuffer>::Adopt(IVulkanAllocatedBuffer::Create(this, desc));
  return rhi::QueryInterfacePtr<IBuffer>(buffer.Get()).Detach();
}

rhi::IComputeProgram *VulkanDeviceImpl::CreateComputeProgram(const ComputeProgramDesc &desc) {
  return IVulkanComputeProgram::Create(this, desc);
}

rhi::IProgram *VulkanDeviceImpl::CreateProgram(const ProgramDesc &desc) {
  return IVulkanProgram::Create(this, desc);
}

rhi::IWindow *VulkanDeviceImpl::CreateGLFWWindow(const WindowDesc &desc) {
  return IVulkanWindow::Create(this, desc);
}

VulkanScratchPool &VulkanDeviceImpl::ScratchPool() {
  return scratch_pool_;
}

IBottomLevelAccelerationStructure *VulkanDeviceImpl::CreateBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructureDesc &desc) {
  return CreateVulkanBottomLevelAccelerationStructure(this, desc);
}

ITopLevelAccelerationStructure *VulkanDeviceImpl::CreateTopLevelAccelerationStructure(
    const TopLevelAccelerationStructureDesc &desc) {
  return IVulkanTopLevelAccelerationStructure::Create(this, desc);
}

IRayTracingProgram *VulkanDeviceImpl::CreateRayTracingProgram(const RayTracingProgramDesc &desc) {
  if (!SupportRayTracingPipeline())
    return nullptr;
  return IVulkanRayTracingProgram::Create(this, desc);
}

VulkanTemporalBufferPool &VulkanDeviceImpl::DownloadPool() {
  return download_pool_;
}

rhi::IDevice *IVulkanDevice::CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                                IVulkanCore *core,
                                                VkPhysicalDevice physical_device,
                                                uint32_t api_version) {
  return VulkanDeviceImpl::CreateVulkanDevice(info, core, physical_device, api_version);
}

void VulkanDeviceImpl::InitializeDevice(PhysicalDevice info, IVulkanCore *core, BackendAPI api) {
  api_ = api;
  core_ = core;
  info_ = std::move(info);
}

rhi::IShader *VulkanDeviceImpl::CreateShader(const ShaderDesc &desc) {
  if (desc.filename.empty() || desc.entry_point.empty() || desc.stage < SHADER_STAGE_VERTEX ||
      desc.stage > SHADER_STAGE_CALLABLE)
    return nullptr;
  return backend::CreateShader(desc, api_);
}

void VulkanDeviceImpl::DestroyDevice() {
  ShutdownQueues();
}

std::string VulkanDeviceImpl::Name() const {
  return info_.name;
}

bool VulkanDeviceImpl::SupportRayTracingPipeline() const {
  return info_.ray_tracing_pipeline_feature;
}

bool VulkanDeviceImpl::SupportRayQuery() const {
  return info_.ray_query_feature;
}

bool VulkanDeviceImpl::SupportGeometryShader() const {
  return info_.geometry_shader_feature;
}

IVulkanQueue *VulkanDeviceImpl::GetNativeQueue(QueueType type) const {
  if (type != QUEUE_TYPE_MAIN && type != QUEUE_TYPE_ASSIST)
    return nullptr;
  return queues_[type].Get();
}

rhi::IQueue *VulkanDeviceImpl::GetQueue(QueueType type) {
  if (type != QUEUE_TYPE_MAIN && type != QUEUE_TYPE_ASSIST)
    return nullptr;
  return rhi::QueryInterfacePtr<IQueue>(queues_[type].Get()).Detach();
}

const PhysicalDevice &VulkanDeviceImpl::GetPhysicalDevice() const noexcept {
  return info_;
}

IVulkanCore *VulkanDeviceImpl::OwnerCore() const {
  return core_;
}

void VulkanDeviceImpl::StartCollector() {
  collector_ = std::thread([this] { Collect(); });
}

bool VulkanDeviceImpl::WaitEvents(const std::vector<Event> &events, bool wait_all) const {
  if (events.empty())
    return wait_all;
  std::vector<Event> unique;
  unique.reserve(events.size());
  for (const auto &event : events) {
    auto timeline = rhi::QueryInterfacePtr<IVulkanTimeline>(event.timeline).Get();
    if (!timeline || timeline->Owner() != this) {
      ReportError(-1, "Host waits require timelines from this device.");
      return false;
    }
    auto found = std::find_if(unique.begin(), unique.end(),
                              [&](const Event &other) { return other.timeline == event.timeline; });
    if (found == unique.end())
      unique.push_back(event);
    else
      found->value = wait_all ? std::max(found->value, event.value) : std::min(found->value, event.value);
  }
  return WaitNativeEvents(unique, wait_all);
}

void VulkanDeviceImpl::Collect() {
  std::vector<Event> next;
  next.reserve(queues_.size());
  for (;;) {
    next.clear();
    bool empty = true;
    for (const auto &queue : queues_) {
      if (!queue)
        continue;
      uint64_t completed = 0;
      auto timeline = rhi::QueryInterfacePtr<ITimeline>(queue->GetTimeline());
      const bool timeline_valid = timeline->GetCompletedValue(completed);
      if (timeline_valid && completed != UINT64_MAX)
        next.push_back({timeline.Get(), completed + 1});
      if (!queue->CollectCompleted(completed, timeline_valid))
        empty = false;
    }
    if (stopping_ && empty)
      return;
    WaitForProgress(next);
  }
}

void VulkanDeviceImpl::ShutdownQueues() {
  if (!collector_.joinable())
    return;
  stopping_ = true;
  InterruptCollector();
  collector_.join();
}
}  // namespace grassland::rhi::backend::vulkan
