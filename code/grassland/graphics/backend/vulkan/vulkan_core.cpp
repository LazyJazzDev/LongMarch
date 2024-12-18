#include "grassland/graphics/backend/vulkan/vulkan_core.h"

#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"
#include "grassland/graphics/backend/vulkan/vulkan_program.h"
#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanCore::VulkanCore(const Settings &settings) : Core(settings) {
  vulkan::InstanceCreateHint hint{};
  hint.SetValidationLayersEnabled(DebugEnabled());
  vulkan::CreateInstance(hint, &instance_);
}

VulkanCore::~VulkanCore() {
}

int VulkanCore::CreateBuffer(size_t size,
                             BufferType type,
                             double_ptr<Buffer> pp_buffer) {
  pp_buffer.construct<VulkanStaticBuffer>(this, size);
  return 0;
}

int VulkanCore::CreateImage(int width,
                            int height,
                            ImageFormat format,
                            double_ptr<Image> pp_image) {
  pp_image.construct<VulkanImage>(this, width, height, format);
  SingleTimeCommand([this, pp_image](VkCommandBuffer command_buffer) {
    VulkanImage *image = dynamic_cast<VulkanImage *>(*pp_image);
    vulkan::TransitImageLayout(
        command_buffer, image->Image()->Handle(), VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_MEMORY_READ_BIT, 0);
  });
  return 0;
}

int VulkanCore::CreateWindowObject(int width,
                                   int height,
                                   const std::string &title,
                                   double_ptr<Window> pp_window) {
  pp_window.construct<VulkanWindow>(this, width, height, title);
  return 0;
}

int VulkanCore::CreateShader(const void *data,
                             size_t size,
                             double_ptr<Shader> pp_shader) {
  pp_shader.construct<VulkanShader>(this, data, size);
  return 0;
}

int VulkanCore::CreateProgram(const std::vector<ImageFormat> &color_formats,
                              ImageFormat depth_format,
                              double_ptr<Program> pp_program) {
  pp_program.construct<VulkanProgram>(this, color_formats, depth_format);
  return 0;
}

int VulkanCore::CreateCommandContext(
    double_ptr<CommandContext> pp_command_context) {
  pp_command_context.construct<VulkanCommandContext>(this);
  return 0;
}

int VulkanCore::SubmitCommandContext(CommandContext *p_command_context) {
  VulkanCommandContext *command_context =
      dynamic_cast<VulkanCommandContext *>(p_command_context);
  for (auto &window : command_context->windows_) {
    window->AcquireNextImage();
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
    vulkan::TransitImageLayout(command_buffer, image, state.layout,
                               VK_IMAGE_LAYOUT_GENERAL, state.stage,
                               VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, state.access,
                               VK_ACCESS_MEMORY_READ_BIT);
  }
  command_context->image_states_.clear();

  vkEndCommandBuffer(command_buffer);

  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkSemaphore> signal_semaphores;

  for (auto &window : command_context->windows_) {
    wait_semaphores.push_back(window->ImageAvailableSemaphore()->Handle());
    signal_semaphores.push_back(window->RenderFinishSemaphore()->Handle());
  }

  VkFence fence = in_flight_fences_[current_frame_]->Handle();
  vkResetFences(device_->Handle(), 1, &fence);

  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  if (!wait_semaphores.empty()) {
    submit_info.waitSemaphoreCount = wait_semaphores.size();
    submit_info.pWaitSemaphores = wait_semaphores.data();
    submit_info.signalSemaphoreCount = signal_semaphores.size();
    submit_info.pSignalSemaphores = signal_semaphores.data();
  }
  vkQueueSubmit(graphics_queue_->Handle(), 1, &submit_info, fence);

  for (auto &window : command_context->windows_) {
    window->Present();
  }

  current_frame_ = (current_frame_ + 1) % FramesInFlight();
  fence = in_flight_fences_[current_frame_]->Handle();
  vkWaitForFences(device_->Handle(), 1, &fence, VK_TRUE,
                  std::numeric_limits<uint64_t>::max());

  return 0;
}

int VulkanCore::GetPhysicalDeviceProperties(
    PhysicalDeviceProperties *p_physical_device_properties) {
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
    if (!physical_device.SupportGeometryShader()) {
      continue;
    }
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
    if (physical_device.SupportGeometryShader()) {
      physical_devices.push_back(physical_device);
    }
  }

  if (device_index < 0 || device_index >= physical_devices.size()) {
    return -1;
  }
  auto physical_device = physical_devices[device_index];
  grassland::vulkan::DeviceFeatureRequirement device_feature_requirement{};
  device_feature_requirement.enable_raytracing_extension =
      physical_device.SupportRayTracing();
  if (instance_->CreateDevice(
          physical_device,
          device_feature_requirement.GenerateRecommendedDeviceCreateInfo(
              physical_device),
          device_feature_requirement.GetVmaAllocatorCreateFlags(),
          &device_) != VK_SUCCESS) {
    return -1;
  }

  device_name_ = physical_device.GetPhysicalDeviceProperties().deviceName;
  ray_tracing_support_ = physical_device.SupportRayTracing();

  vulkan::ThrowIfFailed(device_->CreateCommandPool(
                            device_->PhysicalDevice().GraphicsFamilyIndex(),
                            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                            &graphics_command_pool_),
                        "failed to create graphics command pool");
  vulkan::ThrowIfFailed(device_->CreateCommandPool(
                            device_->PhysicalDevice().TransferFamilyIndex(),
                            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                            &transfer_command_pool_),
                        "failed to create transfer command pool");

  device_->GetQueue(device_->PhysicalDevice().GraphicsFamilyIndex(), 0,
                    &graphics_queue_);
  device_->GetQueue(device_->PhysicalDevice().TransferFamilyIndex(), 0,
                    &transfer_queue_);

  in_flight_fences_.resize(FramesInFlight());
  command_buffers_.resize(FramesInFlight());

  for (int i = 0; i < FramesInFlight(); i++) {
    device_->CreateFence(true, &in_flight_fences_[i]);
    graphics_command_pool_->AllocateCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, &command_buffers_[i]);
  }

  return 0;
}

void VulkanCore::WaitGPU() {
  device_->WaitIdle();
}

void VulkanCore::SingleTimeCommand(
    std::function<void(VkCommandBuffer)> command) {
  vulkan::SingleTimeCommand(graphics_queue_.get(), graphics_command_pool_.get(),
                            command);
}

}  // namespace grassland::graphics::backend
