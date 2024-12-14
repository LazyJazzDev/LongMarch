#include "grassland/graphics/backend/vulkan/vulkan_core.h"

#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
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

  render_finished_semaphores_.resize(FramesInFlight());
  in_flight_fences_.resize(FramesInFlight());
  command_buffers_.resize(FramesInFlight());

  for (int i = 0; i < FramesInFlight(); i++) {
    device_->CreateSemaphore(&render_finished_semaphores_[i]);
    device_->CreateFence(true, &in_flight_fences_[i]);
    graphics_command_pool_->AllocateCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, &command_buffers_[i]);
  }

  return 0;
}

void VulkanCore::WaitGPU() {
  device_->WaitIdle();
}

}  // namespace grassland::graphics::backend
