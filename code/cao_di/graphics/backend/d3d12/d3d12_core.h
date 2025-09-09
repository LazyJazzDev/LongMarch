#pragma once
#include "cao_di/graphics/backend/d3d12/d3d12_util.h"

namespace CD::graphics::backend {

struct BlitPipeline {
  d3d12::Device *device_;
  std::unique_ptr<d3d12::ShaderModule> vertex_shader;
  std::unique_ptr<d3d12::ShaderModule> pixel_shader;
  std::unique_ptr<d3d12::RootSignature> root_signature;
  std::map<DXGI_FORMAT, std::unique_ptr<d3d12::PipelineState>> pipeline_states;
  void Initialize(d3d12::Device *device);
  d3d12::PipelineState *GetPipelineState(DXGI_FORMAT format);
};

class D3D12Core : public Core {
 public:
  D3D12Core(const Settings &settings);
  ~D3D12Core() override;

  BackendAPI API() const override {
    return BACKEND_API_D3D12;
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

  d3d12::DXGIFactory *DXGIFactory() const {
    return dxgi_factory_.get();
  }

  d3d12::Device *Device() const {
    return device_.get();
  }

  d3d12::CommandQueue *CommandQueue() const {
    return command_queue_.get();
  }

  d3d12::CommandList *CommandList() const {
    return command_lists_[current_frame_].get();
  }

  d3d12::CommandAllocator *CommandAllocator() const {
    return command_allocators_[current_frame_].get();
  }

  d3d12::Fence *Fence() const {
    return fence_.get();
  }

  d3d12::CommandAllocator *SingleTimeCommandAllocator() const {
    return single_time_allocator_.get();
  }

  uint32_t CurrentFrame() const override {
    return current_frame_;
  }

  void SingleTimeCommand(std::function<void(ID3D12GraphicsCommandList *)> command);

  BlitPipeline *BlitPipeline() {
    return &blit_pipeline_;
  }

  d3d12::DescriptorHeap *RTVDescriptorHeap() const {
    return rtv_descriptor_heaps_[current_frame_].get();
  }

  d3d12::DescriptorHeap *DSVDescriptorHeap() const {
    return dsv_descriptor_heaps_[current_frame_].get();
  }

  d3d12::Buffer *RequestUploadStagingBuffer(size_t size);
  d3d12::Buffer *RequestDownloadStagingBuffer(size_t size);

#if defined(LONGMARCH_CUDA_RUNTIME)
  void ImportCudaExternalMemory(cudaExternalMemory_t &cuda_memory, d3d12::Buffer *buffer);
  void CUDABeginExecutionBarrier(cudaStream_t stream) override;
  void CUDAEndExecutionBarrier(cudaStream_t stream) override;
#endif

 private:
  std::unique_ptr<d3d12::DXGIFactory> dxgi_factory_;
  std::unique_ptr<d3d12::Device> device_;

  struct BlitPipeline blit_pipeline_;

  std::unique_ptr<d3d12::CommandQueue> command_queue_;
  std::unique_ptr<d3d12::CommandQueue> transfer_command_queue_;
  std::vector<std::unique_ptr<d3d12::CommandAllocator>> command_allocators_;
  std::vector<std::unique_ptr<d3d12::CommandList>> command_lists_;

  std::unique_ptr<d3d12::Fence> fence_;
  std::vector<uint64_t> in_flight_values_;

  std::unique_ptr<d3d12::CommandAllocator> single_time_allocator_;
  std::unique_ptr<d3d12::CommandList> single_time_command_list_;

  std::unique_ptr<d3d12::CommandAllocator> transfer_allocator_;
  std::unique_ptr<d3d12::CommandList> transfer_command_list_;

  std::vector<std::unique_ptr<d3d12::DescriptorHeap>> resource_descriptor_heaps_;
  std::vector<std::unique_ptr<d3d12::DescriptorHeap>> sampler_descriptor_heaps_;

  std::vector<std::unique_ptr<d3d12::DescriptorHeap>> rtv_descriptor_heaps_;
  std::vector<std::unique_ptr<d3d12::DescriptorHeap>> dsv_descriptor_heaps_;

  uint32_t current_frame_{0};

  std::vector<std::vector<std::function<void()>>> post_execute_functions_;

#if defined(LONGMARCH_CUDA_RUNTIME)
  uint32_t cuda_device_node_mask_;
  cudaExternalSemaphore_t cuda_semaphore_{};
#endif

  std::unique_ptr<d3d12::Buffer> upload_staging_buffer_;
  std::unique_ptr<d3d12::Buffer> download_staging_buffer_;
};

}  // namespace CD::graphics::backend
