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

  int CreateWindowObject(int width,
                         int height,
                         const std::string &title,
                         double_ptr<Window> pp_window) override;

  int GetPhysicalDeviceProperties(
      PhysicalDeviceProperties *p_physical_device_properties =
          nullptr) override;

  int InitializeLogicalDevice(int device_index) override;

  vulkan::Instance *Instance() {
    return instance_.get();
  }

  vulkan::Device *Device() {
    return device_.get();
  }

 private:
  std::unique_ptr<vulkan::Instance> instance_;
  std::unique_ptr<vulkan::Device> device_;

  uint32_t current_frame_{0};
  std::vector<std::unique_ptr<vulkan::Semaphore>> render_finished_semaphores_;
  std::vector<std::unique_ptr<vulkan::Fence>> in_flight_fences_;

  std::unique_ptr<vulkan::CommandPool> graphics_command_pool_;
  std::unique_ptr<vulkan::CommandPool> transfer_command_pool_;
  std::vector<std::unique_ptr<vulkan::CommandBuffer>> command_buffers_;

  std::unique_ptr<vulkan::Queue> graphics_queue_;
  std::unique_ptr<vulkan::Queue> transfer_queue_;
};

}  // namespace grassland::graphics::backend
