#include "grassland/graphics/backend/vulkan/vulkan_core.h"

#include "grassland/graphics/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"
#include "grassland/graphics/backend/vulkan/vulkan_program.h"
#include "grassland/graphics/backend/vulkan/vulkan_sampler.h"
#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanCore::VulkanCore(const Settings &settings) : Core(settings) {
  vulkan::InstanceCreateHint hint{};
  hint.SetValidationLayersEnabled(DebugEnabled());
#if defined(LONGMARCH_CUDA_RUNTIME)
  hint.AddExtension(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  hint.AddExtension(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
#endif
  vulkan::CreateInstance(hint, &instance_);
}

VulkanCore::~VulkanCore() {
#if defined(LONGMARCH_CUDA_RUNTIME)
  if (cuda_synchronization_semaphore_) {
    VkSemaphoreWaitInfo semaphore_wait_info{};
    semaphore_wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    semaphore_wait_info.pSemaphores = &cuda_synchronization_semaphore_;
    semaphore_wait_info.pValues = &cuda_synchronization_value_;
    semaphore_wait_info.semaphoreCount = 1;
    vkWaitSemaphores(device_->Handle(), &semaphore_wait_info, std::numeric_limits<uint64_t>::max());
    cudaDestroyExternalSemaphore(cuda_external_semaphore_);
    vkDestroySemaphore(device_->Handle(), cuda_synchronization_semaphore_, nullptr);
  }
#endif

  for (auto &descriptor_set : descriptor_sets_) {
    while (!descriptor_set.empty()) {
      delete descriptor_set.front();
      descriptor_set.pop();
    }
  }
  descriptor_pools_.clear();
}

int VulkanCore::CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) {
  switch (type) {
    case BUFFER_TYPE_DYNAMIC:
      pp_buffer.construct<VulkanDynamicBuffer>(this, size);
      break;
    default:
      pp_buffer.construct<VulkanStaticBuffer>(this, size);
      break;
  }
  return 0;
}

#if defined(LONGMARCH_CUDA_RUNTIME)
int VulkanCore::CreateCUDABuffer(size_t size, double_ptr<CUDABuffer> pp_buffer) {
  pp_buffer.construct<VulkanCUDABuffer>(this, size);
  return 0;
}
#endif

int VulkanCore::CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) {
  pp_image.construct<VulkanImage>(this, width, height, format);
  SingleTimeCommand([this, pp_image](VkCommandBuffer command_buffer) {
    VulkanImage *image = dynamic_cast<VulkanImage *>(*pp_image);
    vulkan::TransitImageLayout(command_buffer, image->Image()->Handle(), VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_MEMORY_READ_BIT, 0,
                               image->Image()->Aspect());
  });
  return 0;
}

int VulkanCore::CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) {
  pp_sampler.construct<VulkanSampler>(this, info);
  return 0;
}

int VulkanCore::CreateWindowObject(int width,
                                   int height,
                                   const std::string &title,
                                   bool fullscreen,
                                   bool resizable,
                                   double_ptr<Window> pp_window) {
  pp_window.construct<VulkanWindow>(this, width, height, title, fullscreen, resizable, false);
  return 0;
}

int VulkanCore::CreateShader(const std::string &source_code,
                             const std::string &entry_point,
                             const std::string &target,
                             double_ptr<Shader> pp_shader) {
  pp_shader.construct<VulkanShader>(
      this,
      CompileShader(source_code, entry_point, target, {"-spirv", "-fspv-target-env=vulkan1.2", "-fvk-use-dx-layout"}));
  return 0;
}

int VulkanCore::CreateProgram(const std::vector<ImageFormat> &color_formats,
                              ImageFormat depth_format,
                              double_ptr<Program> pp_program) {
  pp_program.construct<VulkanProgram>(this, color_formats, depth_format);
  return 0;
}

int VulkanCore::CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) {
  pp_program.construct<VulkanComputeProgram>(this, dynamic_cast<VulkanShader *>(compute_shader));
  return 0;
}

int VulkanCore::CreateCommandContext(double_ptr<CommandContext> pp_command_context) {
  pp_command_context.construct<VulkanCommandContext>(this);
  return 0;
}

int VulkanCore::CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                                       Buffer *index_buffer,
                                                       uint32_t stride,
                                                       double_ptr<AccelerationStructure> pp_blas) {
  VulkanBuffer *vk_vertex_buffer = dynamic_cast<VulkanBuffer *>(vertex_buffer);
  VulkanBuffer *vk_index_buffer = dynamic_cast<VulkanBuffer *>(index_buffer);
  assert(vk_vertex_buffer != nullptr);
  assert(vk_index_buffer != nullptr);
  std::unique_ptr<vulkan::AccelerationStructure> blas;
  device_->CreateBottomLevelAccelerationStructure(
      vk_vertex_buffer->DeviceAddress(), vk_index_buffer->DeviceAddress(), vertex_buffer->Size() / stride, stride,
      index_buffer->Size() / (sizeof(uint32_t) * 3), graphics_command_pool_.get(), graphics_queue_.get(), &blas);
  pp_blas.construct<VulkanAccelerationStructure>(this, std::move(blas));
  return 0;
}

int VulkanCore::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                                    double_ptr<AccelerationStructure> pp_tlas) {
  std::vector<VkAccelerationStructureInstanceKHR> vk_instances;
  vk_instances.reserve(instances.size());
  for (const auto &instance : instances) {
    vk_instances.emplace_back(RayTracingInstanceToVkAccelerationStructureInstanceKHR(instance));
  }
  std::unique_ptr<vulkan::AccelerationStructure> tlas;
  device_->CreateTopLevelAccelerationStructure(vk_instances, graphics_command_pool_.get(), graphics_queue_.get(),
                                               &tlas);
  pp_tlas.construct<VulkanAccelerationStructure>(this, std::move(tlas));
  return 0;
}

int VulkanCore::CreateRayTracingProgram(Shader *raygen_shader,
                                        Shader *miss_shader,
                                        Shader *closest_shader,
                                        double_ptr<RayTracingProgram> pp_program) {
  VulkanShader *vk_raygen_shader = dynamic_cast<VulkanShader *>(raygen_shader);
  VulkanShader *vk_miss_shader = dynamic_cast<VulkanShader *>(miss_shader);
  VulkanShader *vk_closest_shader = dynamic_cast<VulkanShader *>(closest_shader);

  assert(vk_raygen_shader != nullptr);
  assert(vk_miss_shader != nullptr);
  assert(vk_closest_shader != nullptr);

  pp_program.construct<VulkanRayTracingProgram>(this, vk_raygen_shader, vk_miss_shader, vk_closest_shader);

  return 0;
}

int VulkanCore::SubmitCommandContext(CommandContext *p_command_context) {
  VulkanCommandContext *command_context = dynamic_cast<VulkanCommandContext *>(p_command_context);

  auto &set_queue = descriptor_sets_[current_frame_];
  auto &pool = descriptor_pools_[current_frame_];

  while (!set_queue.empty()) {
    delete set_queue.front();
    set_queue.pop();
  }

  auto pool_size = pool->PoolSize();
  auto max_sets = pool->MaxSets();
  bool update_pool = false;
  for (auto &[type, count] : command_context->required_pool_size_.descriptor_type_count) {
    if (!pool_size.descriptor_type_count.count(type) || pool_size.descriptor_type_count[type] < count) {
      uint32_t type_count = pool_size.descriptor_type_count[type];
      if (!type_count) {
        type_count = 1;
      }
      while (type_count < count) {
        type_count *= 2;
      }
      pool_size.descriptor_type_count[type] = type_count;
      update_pool = true;
    }
  }
  while (max_sets < command_context->required_set_count_) {
    max_sets *= 2;
    update_pool = true;
  }
  if (update_pool) {
    pool.reset();
    device_->CreateDescriptorPool(pool_size.ToVkDescriptorPoolSize(), max_sets, &pool);
  }
  current_descriptor_pool_ = pool.get();
  current_descriptor_set_queue_ = &set_queue;

  for (auto &window : command_context->windows_) {
    window->AcquireNextImage();
  }

  if (command_context->dynamic_buffers_.size()) {
    vkResetCommandBuffer(transfer_command_buffer_->Handle(), 0);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(transfer_command_buffer_->Handle(), &begin_info);
    for (auto &buffer : command_context->dynamic_buffers_) {
      buffer->TransferData(transfer_command_buffer_->Handle());
    }
    vkEndCommandBuffer(transfer_command_buffer_->Handle());

    VkCommandBuffer command_buffers[] = {transfer_command_buffer_->Handle()};

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = command_buffers;

#if defined(LONGMARCH_CUDA_RUNTIME)
    VkTimelineSemaphoreSubmitInfo timeline_info = {};
    uint64_t wait_value;
    uint64_t signal_value;
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    if (cuda_device_ >= 0) {
      wait_value = cuda_synchronization_value_++;
      signal_value = cuda_synchronization_value_;
      timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
      timeline_info.waitSemaphoreValueCount = 1;
      timeline_info.pWaitSemaphoreValues = &wait_value;
      timeline_info.signalSemaphoreValueCount = 1;
      timeline_info.pSignalSemaphoreValues = &signal_value;
      timeline_info.pNext = submit_info.pNext;
      submit_info.pNext = &timeline_info;
      submit_info.signalSemaphoreCount = 1;
      submit_info.pSignalSemaphores = &cuda_synchronization_semaphore_;
      submit_info.pWaitDstStageMask = wait_stages;
      submit_info.waitSemaphoreCount = 1;
      submit_info.pWaitSemaphores = &cuda_synchronization_semaphore_;
    }
#endif

    vkQueueSubmit(transfer_queue_->Handle(), 1, &submit_info, nullptr);
  }

  VkCommandBuffer command_buffer = command_buffers_[current_frame_]->Handle();
  vkResetCommandBuffer(command_buffer, 0);
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
    return -1;
  }

  for (auto &command : command_context->commands_) {
    command->CompileCommand(command_context, command_buffer);
  }

  for (auto [image, state] : command_context->image_states_) {
    vulkan::TransitImageLayout(command_buffer, image, state.layout, VK_IMAGE_LAYOUT_GENERAL, state.stage,
                               VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, state.access, VK_ACCESS_MEMORY_READ_BIT,
                               state.aspect);
  }
  command_context->image_states_.clear();

  vkEndCommandBuffer(command_buffer);

  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkSemaphore> signal_semaphores;

  std::vector<VkPipelineStageFlags> wait_stages{};
  for (auto &window : command_context->windows_) {
    wait_semaphores.push_back(window->ImageAvailableSemaphore()->Handle());
    wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    signal_semaphores.push_back(window->RenderFinishSemaphore()->Handle());
  }

  VkFence fence = in_flight_fences_[current_frame_]->Handle();
  vkResetFences(device_->Handle(), 1, &fence);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

#if defined(LONGMARCH_CUDA_RUNTIME)
  VkTimelineSemaphoreSubmitInfo timeline_info = {};
  std::vector<uint64_t> wait_values(wait_semaphores.size() + 1, 0);
  std::vector<uint64_t> signal_values(signal_semaphores.size() + 1, 0);
  if (cuda_device_ >= 0) {
    wait_values[wait_semaphores.size()] = cuda_synchronization_value_++;
    signal_values[signal_semaphores.size()] = cuda_synchronization_value_;
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = wait_values.size();
    timeline_info.pWaitSemaphoreValues = wait_values.data();
    timeline_info.signalSemaphoreValueCount = signal_values.size();
    timeline_info.pSignalSemaphoreValues = signal_values.data();
    timeline_info.pNext = submit_info.pNext;
    submit_info.pNext = &timeline_info;
    wait_semaphores.push_back(cuda_synchronization_semaphore_);
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    signal_semaphores.push_back(cuda_synchronization_semaphore_);
  }
#endif

  if (!wait_semaphores.empty()) {
    submit_info.waitSemaphoreCount = wait_semaphores.size();
    submit_info.pWaitSemaphores = wait_semaphores.data();
    submit_info.pWaitDstStageMask = wait_stages.data();
    submit_info.signalSemaphoreCount = signal_semaphores.size();
    submit_info.pSignalSemaphores = signal_semaphores.data();
  }

  vkQueueSubmit(graphics_queue_->Handle(), 1, &submit_info, fence);

  for (auto &window : command_context->windows_) {
    window->Present();
  }

  post_execute_functions_[current_frame_] = p_command_context->GetPostExecutionCallbacks();

  current_frame_ = (current_frame_ + 1) % FramesInFlight();
  fence = in_flight_fences_[current_frame_]->Handle();
  vkWaitForFences(device_->Handle(), 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());

  vkQueueWaitIdle(transfer_queue_->Handle());

  for (auto &callback : post_execute_functions_[current_frame_]) {
    callback();
  }
  post_execute_functions_[current_frame_].clear();

  return 0;
}

int VulkanCore::GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties) {
  auto physical_devices = instance_->EnumeratePhysicalDevices();
  if (physical_devices.empty()) {
    return 0;
  }
  int num_device = 0;

  for (int i = 0; i < physical_devices.size(); ++i) {
    auto physical_device = physical_devices[i];
    PhysicalDeviceProperties properties{};
    properties.name = physical_device.GetPhysicalDeviceProperties().deviceName;
    properties.score = physical_device.Evaluate();
    properties.ray_tracing_support = physical_device.SupportRayTracing();
    properties.geometry_shader_support = physical_device.SupportGeometryShader();
    if (p_physical_device_properties) {
      p_physical_device_properties[i] = properties;
    }
    num_device++;
  }

  return num_device;
}

int VulkanCore::InitializeLogicalDevice(int device_index) {
  auto original_physical_devices = instance_->EnumeratePhysicalDevices();
  std::vector<vulkan::PhysicalDevice> physical_devices;
  // Erase non geometry shader support devices
  for (auto &physical_device : original_physical_devices) {
    physical_devices.push_back(physical_device);
  }

  if (device_index < 0 || device_index >= physical_devices.size()) {
    return -1;
  }
  auto physical_device = physical_devices[device_index];

#if defined(LONGMARCH_CUDA_RUNTIME)
  VkPhysicalDeviceIDProperties id_properties{};
  id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  id_properties.pNext = nullptr;

  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &id_properties;
  vkGetPhysicalDeviceProperties2(physical_device.Handle(), &properties2);

  int cuda_device_count = 0;
  cudaGetDeviceCount(&cuda_device_count);
  for (int i = 0; i < cuda_device_count; i++) {
    cudaDeviceProp device_properties{};
    cudaGetDeviceProperties(&device_properties, i);
    if (std::memcmp(id_properties.deviceUUID, &device_properties.uuid, VK_UUID_SIZE) == 0) {
      cuda_device_ = i;
      break;
    }
  }
#endif

  grassland::vulkan::DeviceFeatureRequirement device_feature_requirement{};
  device_feature_requirement.enable_raytracing_extension = physical_device.SupportRayTracing();
  auto create_info = device_feature_requirement.GenerateRecommendedDeviceCreateInfo(physical_device);
#if defined(LONGMARCH_CUDA_RUNTIME)
  if (cuda_device_ >= 0) {
    create_info.AddExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    create_info.AddExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    create_info.AddExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN32
    create_info.AddExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    create_info.AddExtension(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    create_info.AddExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    create_info.AddExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif /* _WIN64 */
    VkPhysicalDeviceTimelineSemaphoreFeatures physical_device_timeline_semaphore_features{};
    physical_device_timeline_semaphore_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    physical_device_timeline_semaphore_features.timelineSemaphore = VK_TRUE;
    create_info.AddFeature(physical_device_timeline_semaphore_features);
  }
#endif
  if (instance_->CreateDevice(physical_device, create_info, device_feature_requirement.GetVmaAllocatorCreateFlags(),
                              &device_) != VK_SUCCESS) {
    return -1;
  }
  memory_properties_ = physical_device.GetPhysicalDeviceMemoryProperties();

  device_name_ = physical_device.GetPhysicalDeviceProperties().deviceName;
  ray_tracing_support_ = physical_device.SupportRayTracing();

  vulkan::ThrowIfFailed(
      device_->CreateCommandPool(device_->PhysicalDevice().GraphicsFamilyIndex(),
                                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, &graphics_command_pool_),
      "failed to create graphics command pool");
  vulkan::ThrowIfFailed(
      device_->CreateCommandPool(device_->PhysicalDevice().TransferFamilyIndex(),
                                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, &transfer_command_pool_),
      "failed to create transfer command pool");

  device_->GetQueue(device_->PhysicalDevice().GraphicsFamilyIndex(), 0, &graphics_queue_);
  device_->GetQueue(device_->PhysicalDevice().TransferFamilyIndex(), 0, &transfer_queue_);

  in_flight_fences_.resize(FramesInFlight());
  command_buffers_.resize(FramesInFlight());

  descriptor_pools_.resize(FramesInFlight());
  descriptor_sets_.resize(FramesInFlight());

  post_execute_functions_.resize(FramesInFlight());

  for (int i = 0; i < FramesInFlight(); i++) {
    device_->CreateFence(true, &in_flight_fences_[i]);
    graphics_command_pool_->AllocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, &command_buffers_[i]);

    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 100},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
    };

    device_->CreateDescriptorPool({{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}}, 1, &descriptor_pools_[i]);
  }
  transfer_command_pool_->AllocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, &transfer_command_buffer_);

#if defined(LONGMARCH_CUDA_RUNTIME)

  if (cuda_device_ >= 0) {
    // Create a timeline semaphore
    auto handle_type = vulkan::GetDefaultExternalSemaphoreHandleType();
    VkSemaphoreCreateInfo semaphore_create_info{};
    semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkExportSemaphoreCreateInfoKHR export_semaphore_create_info{};
    export_semaphore_create_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;

    VkSemaphoreTypeCreateInfo export_semaphore_create_info_timeline{};
    export_semaphore_create_info_timeline.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    export_semaphore_create_info_timeline.initialValue = 0;
    export_semaphore_create_info_timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    export_semaphore_create_info.pNext = &export_semaphore_create_info_timeline;
    export_semaphore_create_info.handleTypes = handle_type;

    semaphore_create_info.pNext = &export_semaphore_create_info;
    vulkan::ThrowIfFailed(
        vkCreateSemaphore(device_->Handle(), &semaphore_create_info, nullptr, &cuda_synchronization_semaphore_),
        "Failed to create CUDA synchronization semaphore");

    cudaExternalSemaphoreHandleDesc external_semaphore_handle_desc = {};

    if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      external_semaphore_handle_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    } else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      external_semaphore_handle_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    } else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      external_semaphore_handle_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    } else {
      throw std::runtime_error("Unknown handle type requested!");
    }

#ifdef _WIN32
    external_semaphore_handle_desc.handle.win32.handle = (HANDLE)GetSemaphoreHandle(cuda_synchronization_semaphore_);
#else
    external_semaphore_handle_desc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
#endif

    external_semaphore_handle_desc.flags = 0;

    cudaImportExternalSemaphore(&cuda_external_semaphore_, &external_semaphore_handle_desc);
  }

#endif

  return 0;
}

void VulkanCore::WaitGPU() {
  device_->WaitIdle();
  for (auto &post_execute : post_execute_functions_) {
    for (auto &callback : post_execute) {
      callback();
    }
    post_execute.clear();
  }
}

void VulkanCore::SingleTimeCommand(std::function<void(VkCommandBuffer)> command) {
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
#if defined(LONGMARCH_CUDA_RUNTIME)
  VkTimelineSemaphoreSubmitInfo timeline_info = {};
  uint64_t wait_value;
  uint64_t signal_value;
  VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
  if (cuda_device_ >= 0) {
    wait_value = cuda_synchronization_value_++;
    signal_value = cuda_synchronization_value_;
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = 1;
    timeline_info.pWaitSemaphoreValues = &wait_value;
    timeline_info.signalSemaphoreValueCount = 1;
    timeline_info.pSignalSemaphoreValues = &signal_value;
    submit_info.pNext = &timeline_info;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &cuda_synchronization_semaphore_;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &cuda_synchronization_semaphore_;
    submit_info.pWaitDstStageMask = wait_stages;
  }
#endif
  vulkan::SingleTimeCommand(graphics_queue_.get(), graphics_command_pool_.get(), command, submit_info);
}

uint32_t VulkanCore::FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) {
  for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; i++) {
    if (type_filter & (1 << i) && (memory_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  return ~0;
}

#if defined(LONGMARCH_CUDA_RUNTIME)

void VulkanCore::ImportCudaExternalMemory(cudaExternalMemory_t &cuda_memory,
                                          VkDeviceMemory &vulkan_memory,
                                          VkDeviceSize size) {
  auto handle_type = vulkan::GetDefaultExternalMemoryHandleType();
  cudaExternalMemoryHandleDesc external_memory_handle_desc = {};

  if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  } else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
  } else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
    external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  } else {
    throw std::runtime_error("Unknown handle type requested!");
  }

  external_memory_handle_desc.size = size;

#ifdef _WIN64
  external_memory_handle_desc.handle.win32.handle = (HANDLE)GetMemoryHandle(vulkan_memory);
#else
  externalMemoryHandleDesc.handle.fd = (int)(uintptr_t)getMemHandle(vkMem, handleType);
#endif

  cudaImportExternalMemory(&cuda_memory, &external_memory_handle_desc);
}

void VulkanCore::CUDABeginExecutionBarrier(cudaStream_t stream) {
  if (cuda_device_ < 0) {
    throw std::runtime_error("Not CUDA device!");
  }

  cudaExternalSemaphoreWaitParams wait_params = {};
  wait_params.flags = 0;
  wait_params.params.fence.value = cuda_synchronization_value_;

  cudaWaitExternalSemaphoresAsync(&cuda_external_semaphore_, &wait_params, 1, stream);
}

void VulkanCore::CUDAEndExecutionBarrier(cudaStream_t stream) {
  if (cuda_device_ < 0) {
    throw std::runtime_error("Not CUDA device!");
  }

  cuda_synchronization_value_++;
  cudaExternalSemaphoreSignalParams signal_params = {};
  signal_params.flags = 0;
  signal_params.params.fence.value = cuda_synchronization_value_;
  cudaSignalExternalSemaphoresAsync(&cuda_external_semaphore_, &signal_params, 1, stream);
}

void *VulkanCore::GetMemoryHandle(VkDeviceMemory memory) {
  VkExternalMemoryHandleTypeFlagBits handle_type = vulkan::GetDefaultExternalMemoryHandleType();
#ifdef _WIN32
  HANDLE handle = 0;

  VkMemoryGetWin32HandleInfoKHR vk_memory_get_win32_handle_info_khr = {};
  vk_memory_get_win32_handle_info_khr.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  vk_memory_get_win32_handle_info_khr.pNext = NULL;
  vk_memory_get_win32_handle_info_khr.memory = memory;
  vk_memory_get_win32_handle_info_khr.handleType = handle_type;

  PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
  fpGetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_->Handle(), "vkGetMemoryWin32HandleKHR");
  if (!fpGetMemoryWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
  }
  if (fpGetMemoryWin32HandleKHR(device_->Handle(), &vk_memory_get_win32_handle_info_khr, &handle) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }
  return (void *)handle;
#else
  int fd = -1;

  VkMemoryGetFdInfoKHR vk_memory_get_fd_info_khr = {};
  vk_memory_get_fd_info_khr.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  vk_memory_get_fd_info_khr.pNext = NULL;
  vk_memory_get_fd_info_khr.memory = memory;
  vk_memory_get_fd_info_khr.handleType = handle_type;

  PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
  fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_->Handle(), "vkGetMemoryFdKHR");
  if (!fpGetMemoryFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
  }
  if (fpGetMemoryFdKHR(device_->Handle(), &vk_memory_get_fd_info_khr, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }
  return (void *)(uintptr_t)fd;
#endif
}

void *VulkanCore::GetSemaphoreHandle(VkSemaphore semaphore) {
  VkExternalSemaphoreHandleTypeFlagBits handle_type = vulkan::GetDefaultExternalSemaphoreHandleType();
#ifdef _WIN32
  HANDLE handle;

  VkSemaphoreGetWin32HandleInfoKHR semaphore_get_win32_handle_info_khr = {};
  semaphore_get_win32_handle_info_khr.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  semaphore_get_win32_handle_info_khr.pNext = nullptr;
  semaphore_get_win32_handle_info_khr.semaphore = semaphore;
  semaphore_get_win32_handle_info_khr.handleType = handle_type;

  PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
  fpGetSemaphoreWin32HandleKHR =
      (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device_->Handle(), "vkGetSemaphoreWin32HandleKHR");
  if (!fpGetSemaphoreWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
  }
  if (fpGetSemaphoreWin32HandleKHR(device_->Handle(), &semaphore_get_win32_handle_info_khr, &handle) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }

  return (void *)handle;
#else
  int fd;

  VkSemaphoreGetFdInfoKHR semaphore_get_fd_info_khr = {};
  semaphore_get_fd_info_khr.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  semaphore_get_fd_info_khr.pNext = nullptr;
  semaphore_get_fd_info_khr.semaphore = semaphore;
  semaphore_get_fd_info_khr.handleType = handle_type;

  PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
  fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device_->Handle(), "vkGetSemaphoreFdKHR");
  if (!fpGetSemaphoreFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
  }
  if (fpGetSemaphoreFdKHR(device_->Handle(), &semaphore_get_fd_info_khr, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }

  return (void *)(uintptr_t)fd;
#endif
}
#endif

}  // namespace grassland::graphics::backend
