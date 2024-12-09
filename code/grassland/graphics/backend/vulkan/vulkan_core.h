#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanCore : public Core {
 public:
  VulkanCore(const Settings &settings);
  ~VulkanCore() override;

  int CreateBuffer(size_t size,
                   BufferType type,
                   double_ptr<Buffer> pp_buffer) override;

  int CreateImage(int width,
                  int height,
                  ImageFormat format,
                  double_ptr<Image> pp_image) override;

  int GetPhysicalDeviceProperties(
      PhysicalDeviceProperties *p_physical_device_properties =
          nullptr) override;

  int InitialLogicalDevice(int device_index) override;

  vulkan::Instance *Instance() {
    return instance_.get();
  }

  vulkan::Device *Device() {
    return device_.get();
  }

 private:
  std::unique_ptr<vulkan::Instance> instance_;
  std::unique_ptr<vulkan::Device> device_;
};

}  // namespace grassland::graphics::backend
