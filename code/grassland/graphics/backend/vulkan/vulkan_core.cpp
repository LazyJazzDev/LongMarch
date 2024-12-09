#include "grassland/graphics/backend/vulkan/vulkan_core.h"

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
  return 0;
}

int VulkanCore::CreateImage(int width,
                            int height,
                            ImageFormat format,
                            double_ptr<Image> pp_image) {
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

int VulkanCore::InitialLogicalDevice(int device_index) {
  auto physical_devices = instance_->EnumeratePhysicalDevices();
  return 0;
}

}  // namespace grassland::graphics::backend
