#pragma once
#include <queue>

#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanCore : public Core {
 public:
  VulkanCore(const Settings &settings);
  ~VulkanCore() override;

  BackendAPI API() const override {
    return BACKEND_API_VULKAN;
  }

  int CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) override;

  int CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) override;

  int CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) override;

  int CreateWindowObject(int width,
                         int height,
                         const std::string &title,
                         bool fullscreen,
                         bool resizable,
                         double_ptr<Window> pp_window) override;

  int CreateShader(const std::string &source_code,
                   const std::string &entry_point,
                   const std::string &target,
                   double_ptr<Shader> pp_shader) override;

  int CreateProgram(const std::vector<ImageFormat> &color_formats,
                    ImageFormat depth_format,
                    double_ptr<Program> pp_program) override;

  int CreateCommandContext(double_ptr<CommandContext> pp_command_context) override;

  int CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                             Buffer *index_buffer,
                                             uint32_t stride,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateTopLevelAccelerationStructure(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
                                          double_ptr<AccelerationStructure> pp_tlas) override;

  int CreateRayTracingProgram(Shader *raygen_shader,
                              Shader *miss_shader,
                              Shader *closest_shader,
                              double_ptr<RayTracingProgram> pp_program) override;

  int SubmitCommandContext(CommandContext *p_command_context) override;

  int GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties = nullptr) override;

  int InitializeLogicalDevice(int device_index) override;

  void WaitGPU() override;

  vulkan::Instance *Instance() const {
    return instance_.get();
  }

  vulkan::Device *Device() const {
    return device_.get();
  }

  vulkan::Queue *GraphicsQueue() const {
    return graphics_queue_.get();
  }

  vulkan::Queue *TransferQueue() const {
    return transfer_queue_.get();
  }

  vulkan::CommandPool *GraphicsCommandPool() const {
    return graphics_command_pool_.get();
  }

  vulkan::CommandPool *TransferCommandPool() const {
    return transfer_command_pool_.get();
  }

  vulkan::CommandBuffer *CommandBuffer() const {
    return command_buffers_[current_frame_].get();
  }

  vulkan::Fence *InFlightFence() const {
    return in_flight_fences_[current_frame_].get();
  }

  uint32_t CurrentFrame() const {
    return current_frame_;
  }

  void SingleTimeCommand(std::function<void(VkCommandBuffer)> command);

 private:
  friend class VulkanCommandContext;
  std::unique_ptr<vulkan::Instance> instance_;
  std::unique_ptr<vulkan::Device> device_;

  uint32_t current_frame_{0};
  std::vector<std::unique_ptr<vulkan::Fence>> in_flight_fences_;

  std::vector<std::unique_ptr<vulkan::DescriptorPool>> descriptor_pools_;
  std::vector<std::queue<vulkan::DescriptorSet *>> descriptor_sets_;
  vulkan::DescriptorPool *current_descriptor_pool_{nullptr};
  std::queue<vulkan::DescriptorSet *> *current_descriptor_set_queue_{nullptr};

  std::unique_ptr<vulkan::CommandPool> graphics_command_pool_;
  std::unique_ptr<vulkan::CommandPool> transfer_command_pool_;
  std::vector<std::unique_ptr<vulkan::CommandBuffer>> command_buffers_;
  std::unique_ptr<vulkan::CommandBuffer> transfer_command_buffer_;

  std::unique_ptr<vulkan::Queue> graphics_queue_;
  std::unique_ptr<vulkan::Queue> transfer_queue_;
};

}  // namespace grassland::graphics::backend
