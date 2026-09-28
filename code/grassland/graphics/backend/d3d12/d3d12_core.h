#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

struct BlitPipeline {
  ID3D12Device *device_;
  CompiledShaderBlob vertex_shader;
  CompiledShaderBlob pixel_shader;
  Microsoft::WRL::ComPtr<ID3D12RootSignature> root_signature;
  std::map<DXGI_FORMAT, Microsoft::WRL::ComPtr<ID3D12PipelineState>> pipeline_states;
  void Initialize(ID3D12Device *device);
  ID3D12PipelineState *GetPipelineState(DXGI_FORMAT format);
};

class D3D12Core : public Core {
 public:
  D3D12Core(const Settings &settings);
  ~D3D12Core() override;

  bool DeviceRayQuerySupport() const override {
    if (!device_)
      return false;
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
    return SUCCEEDED(device_.Get()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options))) &&
           options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
  }

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

  HRESULT BuildBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS aabb_buffer,
                                                uint32_t stride,
                                                uint32_t num_aabb,
                                                D3D12_RAYTRACING_GEOMETRY_FLAGS flags,
                                                ID3D12CommandQueue *queue,
                                                ID3D12CommandAllocator *allocator,
                                                Microsoft::WRL::ComPtr<ID3D12Resource> &result);

  HRESULT BuildBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS vertex_buffer,
                                                D3D12_GPU_VIRTUAL_ADDRESS index_buffer,
                                                uint32_t num_vertex,
                                                uint32_t stride,
                                                uint32_t primitive_count,
                                                D3D12_RAYTRACING_GEOMETRY_FLAGS flags,
                                                ID3D12CommandQueue *queue,
                                                ID3D12CommandAllocator *allocator,
                                                Microsoft::WRL::ComPtr<ID3D12Resource> &result);

  HRESULT BuildTopLevelAccelerationStructure(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC> &instances,
                                             ID3D12CommandQueue *queue,
                                             ID3D12CommandAllocator *allocator,
                                             Microsoft::WRL::ComPtr<ID3D12Resource> &result);

  HRESULT CreateRayTracingPipeline(ID3D12RootSignature *root_signature,
                                   const CompiledShaderBlob *ray_gen_shader,
                                   const std::vector<const CompiledShaderBlob *> &miss_shaders,
                                   const std::vector<d3d12::HitGroup> &hit_groups,
                                   const std::vector<const CompiledShaderBlob *> &callable_shaders,
                                   Microsoft::WRL::ComPtr<ID3D12StateObject> &pipeline);

  HRESULT CreateShaderTable(ID3D12StateObject *pipeline,
                            const std::vector<int32_t> &miss_shader_indices,
                            const std::vector<int32_t> &hit_group_indices,
                            const std::vector<int32_t> &callable_shader_indices,
                            Microsoft::WRL::ComPtr<ID3D12Resource> &buffer,
                            D3D12_GPU_VIRTUAL_ADDRESS &miss_offset,
                            D3D12_GPU_VIRTUAL_ADDRESS &hit_group_offset,
                            D3D12_GPU_VIRTUAL_ADDRESS &callable_offset) const;

  int SubmitCommandContext(CommandContext *p_command_context) override;

  int GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties = nullptr) override;

  int InitializeLogicalDevice(int device_index) override;

  void WaitGPU() override;

  uint32_t WaveSize() const override;

  IDXGIFactory4 *DXGIFactory() const {
    return dxgi_factory_.Get();
  }

  ID3D12Device *Device() const {
    return device_.Get();
  }

  ID3D12Device5 *DXRDevice() const {
    return dxr_device_.Get();
  }

  ID3D12CommandQueue *CommandQueue() const {
    return command_queue_.Get();
  }

  ID3D12GraphicsCommandList *CommandList() const {
    return command_lists_[current_frame_].Get();
  }

  ID3D12CommandAllocator *CommandAllocator() const {
    return command_allocators_[current_frame_].Get();
  }

  ID3D12Fence *Fence() const {
    return fence_.Get();
  }

  ID3D12CommandAllocator *SingleTimeCommandAllocator() const {
    return single_time_allocator_.Get();
  }

  uint32_t CurrentFrame() const override {
    return current_frame_;
  }

  void SingleTimeCommand(std::function<void(ID3D12GraphicsCommandList *)> command);

  BlitPipeline *BlitPipeline() {
    return &blit_pipeline_;
  }

  CD3DX12_CPU_DESCRIPTOR_HANDLE RTVDescriptorHandle(uint32_t index) const;
  CD3DX12_CPU_DESCRIPTOR_HANDLE DSVDescriptorHandle(uint32_t index) const;

  ID3D12Resource *RequestUploadStagingBuffer(size_t size);
  ID3D12Resource *RequestDownloadStagingBuffer(size_t size);

#if defined(LONGMARCH_CUDA_RUNTIME)
  void ImportCudaExternalMemory(cudaExternalMemory_t &cuda_memory, ID3D12Resource *buffer);
  void CUDABeginExecutionBarrier(cudaStream_t stream) override;
  void CUDAEndExecutionBarrier(cudaStream_t stream) override;
#endif

 private:
  friend class D3D12AccelerationStructure;
  Microsoft::WRL::ComPtr<IDXGIFactory4> dxgi_factory_;
  Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter_;
  Microsoft::WRL::ComPtr<ID3D12Device> device_;
  Microsoft::WRL::ComPtr<ID3D12Device5> dxr_device_;
  D3D12_FEATURE_DATA_D3D12_OPTIONS1 d3d12_options1_{};
  Microsoft::WRL::ComPtr<ID3D12Resource> scratch_buffer_;
  Microsoft::WRL::ComPtr<ID3D12Resource> instance_buffer_;
  ID3D12Resource *RequestScratchBuffer(size_t size);
  ID3D12Resource *RequestInstanceBuffer(size_t size);

  struct BlitPipeline blit_pipeline_;

  Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue_;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> transfer_command_queue_;
  std::vector<Microsoft::WRL::ComPtr<ID3D12CommandAllocator>> command_allocators_;
  std::vector<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>> command_lists_;

  Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
  uint64_t fence_value_{1};
  HANDLE fence_event_{nullptr};
  void SignalFence(ID3D12CommandQueue *queue);
  void QueueWaitFence(ID3D12CommandQueue *queue);
  void WaitForFence(uint64_t value);
  std::vector<uint64_t> in_flight_values_;

  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> single_time_allocator_;
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> single_time_command_list_;

  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> transfer_allocator_;
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> transfer_command_list_;

  std::vector<Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>> resource_descriptor_heaps_;
  std::vector<Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>> sampler_descriptor_heaps_;

  std::vector<Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>> rtv_descriptor_heaps_;
  std::vector<Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>> dsv_descriptor_heaps_;

  uint32_t current_frame_{0};

  std::vector<std::vector<std::function<void()>>> post_execute_functions_;

#if defined(LONGMARCH_CUDA_RUNTIME)
  uint32_t cuda_device_node_mask_;
  cudaExternalSemaphore_t cuda_semaphore_{};
#endif

  Microsoft::WRL::ComPtr<ID3D12Resource> upload_staging_buffer_;
  Microsoft::WRL::ComPtr<ID3D12Resource> download_staging_buffer_;
};

}  // namespace grassland::graphics::backend
