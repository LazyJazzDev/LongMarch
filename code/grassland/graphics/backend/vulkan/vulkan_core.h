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

#if defined(LONGMARCH_CUDA_RUNTIME)
  int CreateCUDABuffer(size_t size, double_ptr<CUDABuffer> pp_buffer) override;
#endif

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

  int CreateShader(const VirtualFileSystem &vfs,
                   const std::string &source_file,
                   const std::string &entry_point,
                   const std::string &target,
                   double_ptr<Shader> pp_shader) override;

  int CreateProgram(const std::vector<ImageFormat> &color_formats,
                    ImageFormat depth_format,
                    double_ptr<Program> pp_program) override;

  int CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) override;

  int CreateCommandContext(double_ptr<CommandContext> pp_command_context) override;

  int CreateBottomLevelAccelerationStructure(Buffer *aabb_buffer,
                                             uint32_t stride,
                                             uint32_t num_aabb,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                             Buffer *index_buffer,
                                             uint32_t num_vertex,
                                             uint32_t stride,
                                             uint32_t num_primitive,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                             Buffer *index_buffer,
                                             uint32_t stride,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                          double_ptr<AccelerationStructure> pp_tlas) override;

  int CreateRayTracingProgram(double_ptr<RayTracingProgram> pp_program) override;

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

  uint32_t CurrentFrame() const override {
    return current_frame_;
  }

  void SingleTimeCommand(std::function<void(VkCommandBuffer)> command);

  uint32_t FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);

#if defined(LONGMARCH_CUDA_RUNTIME)
  void ImportCudaExternalMemory(cudaExternalMemory_t &cuda_memory, VkDeviceMemory &vulkan_memory, VkDeviceSize size);
  void CUDABeginExecutionBarrier(cudaStream_t stream) override;
  void CUDAEndExecutionBarrier(cudaStream_t stream) override;
#endif

 private:
  friend class VulkanCommandContext;
  std::unique_ptr<vulkan::Instance> instance_;
  std::unique_ptr<vulkan::Device> device_;
  VkPhysicalDeviceMemoryProperties memory_properties_;

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

  std::vector<std::vector<std::function<void()>>> post_execute_functions_;

#if defined(LONGMARCH_CUDA_RUNTIME)
  VkSemaphore cuda_synchronization_semaphore_{VK_NULL_HANDLE};
  uint64_t cuda_synchronization_value_{0};
  cudaExternalSemaphore_t cuda_external_semaphore_{nullptr};

  void *GetMemoryHandle(VkDeviceMemory memory);
  void *GetSemaphoreHandle(VkSemaphore semaphore);
#endif
};

}  // namespace grassland::graphics::backend
