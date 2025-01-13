#pragma once
#include "grassland/d3d12/adapter.h"

namespace grassland::d3d12 {
class Device {
 public:
  Device(const Adapter &adapter, D3D_FEATURE_LEVEL feature_level, ComPtr<ID3D12Device> device);

  ID3D12Device *Handle() const {
    return device_.Get();
  }

  const Adapter &Adapter() const {
    return adapter_;
  }

  D3D_FEATURE_LEVEL FeatureLevel() const {
    return feature_level_;
  }

  HRESULT CreateCommandQueue(D3D12_COMMAND_LIST_TYPE type, double_ptr<CommandQueue> pp_command_queue);

  HRESULT CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, double_ptr<CommandAllocator> pp_command_allocator);

  HRESULT CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC &desc, double_ptr<DescriptorHeap> pp_descriptor_heap);

  HRESULT CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type,
                               uint32_t num_descriptors,
                               double_ptr<DescriptorHeap> pp_descriptor_heap);

  HRESULT CreateFence(double_ptr<Fence> pp_fence);

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

  HRESULT CreateShaderModule(const void *compiled_shader_data, size_t size, double_ptr<ShaderModule> pp_shader_module);

  HRESULT CreateShaderModule(const CompiledShaderBlob &compiled_shader, double_ptr<ShaderModule> pp_shader_module);

  HRESULT CreateRootSignature(const CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC &desc,
                              double_ptr<RootSignature> pp_root_signature);

  HRESULT CreatePipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &desc,
                              double_ptr<PipelineState> pp_pipeline_state);

  HRESULT CreateBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS vertex_buffer,
                                                 D3D12_GPU_VIRTUAL_ADDRESS index_buffer,
                                                 uint32_t num_vertex,
                                                 uint32_t stride,
                                                 uint32_t primitive_count,
                                                 CommandQueue *queue,
                                                 Fence *fence,
                                                 CommandAllocator *allocator,
                                                 double_ptr<AccelerationStructure> pp_as);

  HRESULT CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                                 Buffer *index_buffer,
                                                 uint32_t stride,
                                                 CommandQueue *queue,
                                                 Fence *fence,
                                                 CommandAllocator *allocator,
                                                 double_ptr<AccelerationStructure> pp_as);

  HRESULT CreateTopLevelAccelerationStructure(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
                                              CommandQueue *queue,
                                              Fence *fence,
                                              CommandAllocator *allocator,
                                              double_ptr<AccelerationStructure> pp_tlas);

 private:
  class Adapter adapter_;
  ComPtr<ID3D12Device> device_;
  D3D_FEATURE_LEVEL feature_level_;

  // Get DXR device
  ComPtr<ID3D12Device5> dxr_device_;
};
}  // namespace grassland::d3d12
