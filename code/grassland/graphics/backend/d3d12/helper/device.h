#pragma once
#include "grassland/graphics/backend/d3d12/helper/d3d12util.h"

namespace grassland::graphics::backend::d3d12 {
class Device {
 public:
  Device(IDXGIAdapter1 *adapter, D3D_FEATURE_LEVEL feature_level, ComPtr<ID3D12Device> device);

  ID3D12Device *Handle() const {
    return device_.Get();
  }

  ID3D12Device5 *DXRDevice() const {
    return dxr_device_.Get();
  }

  IDXGIAdapter1 *Adapter() const {
    return adapter_.Get();
  }

  D3D_FEATURE_LEVEL FeatureLevel() const {
    return feature_level_;
  }

  UINT WaveLaneCountMax() const {
    return d3d12_options1_.WaveLaneCountMax;
  }

  HRESULT CreateBuffer(size_t size,
                       D3D12_HEAP_TYPE heap_type,
                       D3D12_HEAP_FLAGS heap_flags,
                       D3D12_RESOURCE_STATES resource_state,
                       D3D12_RESOURCE_FLAGS resource_flags,
                       double_ptr<Buffer> pp_buffer);

  HRESULT CreateBuffer(size_t size,
                       D3D12_HEAP_TYPE heap_type,
                       D3D12_RESOURCE_STATES resource_state,
                       D3D12_RESOURCE_FLAGS resource_flags,
                       double_ptr<Buffer> pp_buffer);

  HRESULT CreateBuffer(size_t size,
                       D3D12_HEAP_TYPE heap_type,
                       D3D12_RESOURCE_STATES resource_state,
                       double_ptr<Buffer> pp_buffer);

  HRESULT CreateBuffer(size_t size, D3D12_HEAP_TYPE heap_type, double_ptr<Buffer> pp_buffer);

  HRESULT CreateBuffer(size_t size, double_ptr<Buffer> pp_buffer);

 private:
  HRESULT CreateImage(const D3D12_RESOURCE_DESC &desc, double_ptr<Image> pp_image);

 public:
  HRESULT CreateImage(size_t width,
                      size_t height,
                      DXGI_FORMAT format,
                      D3D12_RESOURCE_FLAGS flags,
                      double_ptr<Image> pp_image);

  HRESULT CreateImage(size_t width, size_t height, DXGI_FORMAT format, double_ptr<Image> pp_image);

  HRESULT CreateImageF32(size_t width, size_t height, double_ptr<Image> pp_image);

  HRESULT CreateImageU8(size_t width, size_t height, double_ptr<Image> pp_image);

  HRESULT CreateBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS aabb_buffer,
                                                 uint32_t stride,
                                                 uint32_t num_aabb,
                                                 D3D12_RAYTRACING_GEOMETRY_FLAGS flags,
                                                 ID3D12CommandQueue *queue,
                                                 ID3D12CommandAllocator *allocator,
                                                 double_ptr<AccelerationStructure> pp_as);

  HRESULT CreateBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS vertex_buffer,
                                                 D3D12_GPU_VIRTUAL_ADDRESS index_buffer,
                                                 uint32_t num_vertex,
                                                 uint32_t stride,
                                                 uint32_t primitive_count,
                                                 D3D12_RAYTRACING_GEOMETRY_FLAGS flags,
                                                 ID3D12CommandQueue *queue,
                                                 ID3D12CommandAllocator *allocator,
                                                 double_ptr<AccelerationStructure> pp_as);

  HRESULT CreateBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS vertex_buffer,
                                                 D3D12_GPU_VIRTUAL_ADDRESS index_buffer,
                                                 uint32_t num_vertex,
                                                 uint32_t stride,
                                                 uint32_t primitive_count,
                                                 ID3D12CommandQueue *queue,
                                                 ID3D12CommandAllocator *allocator,
                                                 double_ptr<AccelerationStructure> pp_as);

  HRESULT CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                                 Buffer *index_buffer,
                                                 uint32_t stride,
                                                 ID3D12CommandQueue *queue,
                                                 ID3D12CommandAllocator *allocator,
                                                 double_ptr<AccelerationStructure> pp_as);

  HRESULT CreateTopLevelAccelerationStructure(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC> &instances,
                                              ID3D12CommandQueue *queue,
                                              ID3D12CommandAllocator *allocator,
                                              double_ptr<AccelerationStructure> pp_tlas);

  HRESULT CreateTopLevelAccelerationStructure(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
                                              ID3D12CommandQueue *queue,
                                              ID3D12CommandAllocator *allocator,
                                              double_ptr<AccelerationStructure> pp_tlas);

  HRESULT CreateRayTracingPipeline(ID3D12RootSignature *root_signature,
                                   const CompiledShaderBlob *ray_gen_shader,
                                   const std::vector<const CompiledShaderBlob *> &miss_shaders,
                                   const std::vector<HitGroup> &hit_groups,
                                   const std::vector<const CompiledShaderBlob *> &callable_shaders,
                                   double_ptr<RayTracingPipeline> pp_pipeline);

  HRESULT CreateRayTracingPipeline(ID3D12RootSignature *root_signature,
                                   const CompiledShaderBlob *ray_gen_shader,
                                   const CompiledShaderBlob *miss_shader,
                                   const CompiledShaderBlob *closest_hit_shader,
                                   double_ptr<RayTracingPipeline> pp_pipeline);

  HRESULT CreateShaderTable(RayTracingPipeline *ray_tracing_pipeline,
                            const std::vector<int32_t> &miss_shader_indices,
                            const std::vector<int32_t> &hit_group_indices,
                            const std::vector<int32_t> &callable_shader_indices,
                            double_ptr<ShaderTable> pp_shader_table) const;

  HRESULT CreateShaderTable(RayTracingPipeline *ray_tracing_pipeline, double_ptr<ShaderTable> pp_shader_table) const;

 private:
  friend AccelerationStructure;

  ID3D12Resource *RequestScratchBuffer(size_t size);
  ID3D12Resource *RequestInstanceBuffer(size_t size);

  ComPtr<IDXGIAdapter1> adapter_;
  ComPtr<ID3D12Device> device_;
  D3D_FEATURE_LEVEL feature_level_;
  D3D12_FEATURE_DATA_D3D12_OPTIONS1 d3d12_options1_;

  // Get DXR device
  ComPtr<ID3D12Device5> dxr_device_;
  ComPtr<ID3D12Resource> scratch_buffer_;
  ComPtr<ID3D12Resource> instance_buffer_;
};
}  // namespace grassland::graphics::backend::d3d12
