#pragma once
#include <queue>

#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanCore : public Core {
 public:
  VulkanCore(const Settings &settings);
  ~VulkanCore() override;

  bool DeviceRayQuerySupport() const override {
    return device_ && physical_device_ && physical_device_->SupportRayQuery();
  }

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

  int CreateShader(const VirtualFileSystem &vfs,
                   const std::string &source_file,
                   const std::string &entry_point,
                   const std::string &target,
                   const std::vector<std::string> &args,
                   double_ptr<Shader> pp_shader) override;

  int CreateProgram(const std::vector<ImageFormat> &color_formats,
                    ImageFormat depth_format,
                    double_ptr<Program> pp_program) override;

  int CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) override;

  int CreateCommandContext(double_ptr<CommandContext> pp_command_context) override;

  int CreateBottomLevelAccelerationStructure(BufferRange aabb_buffer,
                                             uint32_t stride,
                                             uint32_t num_aabb,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateBottomLevelAccelerationStructure(BufferRange vertex_buffer,
                                             BufferRange index_buffer,
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

  uint32_t WaveSize() const override;

  VkInstance Instance() const {
    return instance_;
  }

  const vulkan::InstanceProcedures &InstanceProcedures() const {
    return instance_procedures_;
  }

  VkDevice Handle() const {
    return device_;
  }

  const vulkan::PhysicalDevice &PhysicalDevice() const {
    return *physical_device_;
  }

  const vulkan::DeviceCreateInfo &CreateInfo() const {
    return *create_info_;
  }

  vulkan::DeviceProcedures &Procedures() {
    return procedures_;
  }

  const vulkan::DeviceProcedures &Procedures() const {
    return procedures_;
  }

  VmaAllocator Allocator() const {
    return allocator_;
  }

  VkResult WaitIdle() const {
    return vkDeviceWaitIdle(device_);
  }

  uint32_t SubGroupSize() const {
    return subgroup_properties_.subgroupSize;
  }

  VkResult CreateDescriptorPool(const std::vector<VkDescriptorPoolSize> &pool_sizes,
                                uint32_t max_sets,
                                VkDescriptorPool *pool) const;

  VkResult CreateImage(VkFormat format,
                       VkExtent2D extent,
                       VkImage *image,
                       VkImageView *view,
                       VmaAllocation *allocation) const;

  VkResult CreateBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        VmaMemoryUsage memory_usage,
                        VmaAllocationCreateFlags flags,
                        VkDeviceSize alignment,
                        VkBuffer *buffer,
                        VmaAllocation *allocation) const;

  VkResult CreateBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        VmaMemoryUsage memory_usage,
                        VkBuffer *buffer,
                        VmaAllocation *allocation) const {
    return CreateBuffer(size, usage, memory_usage, 0, 0, buffer, allocation);
  }

  VkDeviceAddress BufferAddress(VkBuffer buffer) const {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device_, &info);
  }

  VkResult CreateBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        VmaMemoryUsage memory_usage,
                        VmaAllocationCreateFlags flags,
                        VkDeviceSize alignment,
                        double_ptr<vulkan::Buffer> pp_buffer) const;

  VkResult CreateBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        VmaMemoryUsage memory_usage,
                        VmaAllocationCreateFlags flags,
                        double_ptr<vulkan::Buffer> pp_buffer) const;

  VkResult CreateBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        VmaMemoryUsage memory_usage,
                        double_ptr<vulkan::Buffer> pp_buffer) const;

  VkResult CreatePipeline(const vulkan::PipelineSettings &settings, VkPipeline *pipeline) const;

  VkResult CreateBottomLevelAccelerationStructure(VkDeviceAddress aabb_address,
                                                  VkDeviceSize stride,
                                                  uint32_t num_aabb,
                                                  VkGeometryFlagsKHR flags,
                                                  VkCommandPool command_pool,
                                                  VkQueue queue,
                                                  double_ptr<vulkan::AccelerationStructure> pp_blas);

  VkResult CreateBottomLevelAccelerationStructure(VkDeviceAddress vertex_buffer_address,
                                                  VkDeviceAddress index_buffer_address,
                                                  uint32_t num_vertex,
                                                  VkDeviceSize stride,
                                                  uint32_t primitive_count,
                                                  VkGeometryFlagsKHR flags,
                                                  VkCommandPool command_pool,
                                                  VkQueue queue,
                                                  double_ptr<vulkan::AccelerationStructure> pp_blas);

  VkResult CreateBottomLevelAccelerationStructure(VkDeviceAddress vertex_buffer_address,
                                                  VkDeviceAddress index_buffer_address,
                                                  uint32_t num_vertex,
                                                  VkDeviceSize stride,
                                                  uint32_t primitive_count,
                                                  VkCommandPool command_pool,
                                                  VkQueue queue,
                                                  double_ptr<vulkan::AccelerationStructure> pp_blas);

  VkResult CreateBottomLevelAccelerationStructure(vulkan::Buffer *vertex_buffer,
                                                  vulkan::Buffer *index_buffer,
                                                  VkDeviceSize stride,
                                                  VkCommandPool command_pool,
                                                  VkQueue queue,
                                                  double_ptr<vulkan::AccelerationStructure> pp_blas);

  VkResult CreateTopLevelAccelerationStructure(const std::vector<VkAccelerationStructureInstanceKHR> &instances,
                                               VkCommandPool command_pool,
                                               VkQueue queue,
                                               double_ptr<vulkan::AccelerationStructure> pp_tlas);

  VkResult CreateTopLevelAccelerationStructure(
      const std::vector<std::pair<vulkan::AccelerationStructure *, glm::mat4>> &objects,
      VkCommandPool command_pool,
      VkQueue queue,
      double_ptr<vulkan::AccelerationStructure> pp_tlas);

  VkResult CreateRayTracingPipeline(VkPipelineLayout pipeline_layout,
                                    VulkanShader *ray_gen_shader,
                                    const std::vector<VulkanShader *> &miss_shaders,
                                    const std::vector<vulkan::HitGroup> &hit_groups,
                                    const std::vector<VulkanShader *> &callable_shaders,
                                    VkPipeline *pipeline) const;
  VkResult CreateShaderBindingTable(VkPipeline pipeline,
                                    size_t miss_shader_count,
                                    size_t hit_group_count,
                                    const std::vector<int32_t> &miss_shader_indices,
                                    const std::vector<int32_t> &hit_group_indices,
                                    const std::vector<int32_t> &callable_shader_indices,
                                    double_ptr<vulkan::ShaderBindingTable> pp_sbt) const;

  VkQueue GraphicsQueue() const {
    return graphics_queue_;
  }

  VkQueue TransferQueue() const {
    return transfer_queue_;
  }

  VkCommandPool GraphicsCommandPool() const {
    return graphics_command_pool_;
  }

  VkCommandPool TransferCommandPool() const {
    return transfer_command_pool_;
  }

  VkCommandBuffer CommandBuffer() const {
    return command_buffers_[current_frame_];
  }

  VkFence InFlightFence() const {
    return in_flight_fences_[current_frame_];
  }

  uint32_t CurrentFrame() const override {
    return current_frame_;
  }

  void SingleTimeCommand(std::function<void(VkCommandBuffer)> command);

  uint32_t FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);

  VkBuffer RequestUploadStagingBuffer(size_t size);
  VkBuffer RequestDownloadStagingBuffer(size_t size);

  VmaAllocation UploadStagingAllocation() const {
    return upload_staging_allocation_;
  }

  VmaAllocation DownloadStagingAllocation() const {
    return download_staging_allocation_;
  }

#if defined(LONGMARCH_CUDA_RUNTIME)
  void ImportCudaExternalMemory(cudaExternalMemory_t &cuda_memory, VkDeviceMemory &vulkan_memory, VkDeviceSize size);
  void CUDABeginExecutionBarrier(cudaStream_t stream) override;
  void CUDAEndExecutionBarrier(cudaStream_t stream) override;
#endif

 private:
  friend class VulkanCommandContext;
  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
  vulkan::InstanceCreateHint instance_hint_{};
  vulkan::InstanceProcedures instance_procedures_{};
  VkDevice device_{VK_NULL_HANDLE};
  uint32_t api_version_{};
  std::optional<vulkan::PhysicalDevice> physical_device_;
  std::optional<vulkan::DeviceCreateInfo> create_info_;
  VkPhysicalDeviceSubgroupProperties subgroup_properties_{};
  vulkan::DeviceProcedures procedures_{};
  VmaAllocator allocator_{VK_NULL_HANDLE};
  void InitializeNativeDevice(uint32_t api_version,
                              const vulkan::PhysicalDevice &physical_device,
                              vulkan::DeviceCreateInfo create_info,
                              VmaAllocatorCreateFlags allocator_flags,
                              VkDevice device);
  void DestroyNativeDevice();
  VkPhysicalDeviceMemoryProperties memory_properties_;

  uint32_t current_frame_{0};
  std::vector<VkFence> in_flight_fences_;

  std::vector<VkDescriptorPool> descriptor_pools_;
  std::vector<vulkan::DescriptorPoolSize> descriptor_pool_sizes_;
  std::vector<uint32_t> descriptor_pool_max_sets_;
  VkDescriptorPool current_descriptor_pool_{VK_NULL_HANDLE};

  VkCommandPool graphics_command_pool_{VK_NULL_HANDLE};
  VkCommandPool transfer_command_pool_{VK_NULL_HANDLE};
  std::vector<VkCommandBuffer> command_buffers_;
  VkCommandBuffer transfer_command_buffer_{VK_NULL_HANDLE};

  VkQueue graphics_queue_{VK_NULL_HANDLE};
  VkQueue transfer_queue_{VK_NULL_HANDLE};

  std::vector<std::vector<std::function<void()>>> post_execute_functions_;

#if defined(LONGMARCH_CUDA_RUNTIME)
  VkSemaphore cuda_synchronization_semaphore_{VK_NULL_HANDLE};
  uint64_t cuda_synchronization_value_{0};
  cudaExternalSemaphore_t cuda_external_semaphore_{nullptr};

  void *GetMemoryHandle(VkDeviceMemory memory);
  void *GetSemaphoreHandle(VkSemaphore semaphore);
#endif

  VkBuffer upload_staging_buffer_{VK_NULL_HANDLE};
  VmaAllocation upload_staging_allocation_{VK_NULL_HANDLE};
  size_t upload_staging_size_{};
  VkBuffer download_staging_buffer_{VK_NULL_HANDLE};
  VmaAllocation download_staging_allocation_{VK_NULL_HANDLE};
  size_t download_staging_size_{};
};

}  // namespace grassland::graphics::backend
