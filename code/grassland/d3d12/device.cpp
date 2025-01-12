#include "grassland/d3d12/device.h"

#include "grassland/d3d12/buffer.h"
#include "grassland/d3d12/command_allocator.h"
#include "grassland/d3d12/command_list.h"
#include "grassland/d3d12/command_queue.h"
#include "grassland/d3d12/descriptor_heap.h"
#include "grassland/d3d12/fence.h"
#include "grassland/d3d12/image.h"
#include "grassland/d3d12/pipeline_state.h"
#include "grassland/d3d12/root_signature.h"
#include "grassland/d3d12/shader_module.h"

namespace grassland::d3d12 {

Device::Device(const class Adapter &adapter, const D3D_FEATURE_LEVEL feature_level, ComPtr<ID3D12Device> device)
    : adapter_(adapter), feature_level_(feature_level), device_(std::move(device)) {
  // Get DXR interfaces
  ThrowIfFailed(device_->QueryInterface(IID_PPV_ARGS(&dxr_device_)), "failed to get DXR device interface.");
}

HRESULT Device::CreateCommandQueue(D3D12_COMMAND_LIST_TYPE type, double_ptr<CommandQueue> pp_command_queue) {
  D3D12_COMMAND_QUEUE_DESC desc = {};
  desc.Type = type;

  ComPtr<ID3D12CommandQueue> command_queue;
  RETURN_IF_FAILED_HR(device_->CreateCommandQueue(&desc, IID_PPV_ARGS(&command_queue)),
                      "failed to create command queue.");

  pp_command_queue.construct(command_queue);
  return S_OK;
}

HRESULT Device::CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type,
                                       double_ptr<CommandAllocator> pp_command_allocator) {
  ComPtr<ID3D12CommandAllocator> command_allocator;
  RETURN_IF_FAILED_HR(device_->CreateCommandAllocator(type, IID_PPV_ARGS(&command_allocator)),
                      "failed to create command allocator.");

  pp_command_allocator.construct(command_allocator);
  return S_OK;
}

HRESULT Device::CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC &desc,
                                     double_ptr<DescriptorHeap> pp_descriptor_heap) {
  ComPtr<ID3D12DescriptorHeap> descriptor_heap;
  RETURN_IF_FAILED_HR(device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&descriptor_heap)),
                      "failed to create descriptor heap.");
  pp_descriptor_heap.construct(descriptor_heap);
  return S_OK;
}

HRESULT Device::CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type,
                                     uint32_t num_descriptors,
                                     double_ptr<DescriptorHeap> pp_descriptor_heap) {
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Type = type;
  desc.NumDescriptors = num_descriptors;
  desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  }

  return CreateDescriptorHeap(desc, pp_descriptor_heap);
}

HRESULT Device::CreateFence(double_ptr<Fence> pp_fence) {
  ComPtr<ID3D12Fence> fence;

  device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

  pp_fence.construct(fence);

  return S_OK;
}

HRESULT Device::CreateBuffer(size_t size,
                             D3D12_HEAP_TYPE heap_type,
                             D3D12_RESOURCE_STATES resource_state,
                             D3D12_RESOURCE_FLAGS resource_flags,
                             double_ptr<Buffer> pp_buffer) {
  ComPtr<ID3D12Resource> buffer;

  RETURN_IF_FAILED_HR(
      ::grassland::d3d12::CreateBuffer(device_.Get(), size, heap_type, resource_state, resource_flags, buffer),
      "failed to create buffer.");

  pp_buffer.construct(buffer);

  return S_OK;
}

HRESULT Device::CreateBuffer(size_t size,
                             D3D12_HEAP_TYPE heap_type,
                             D3D12_RESOURCE_STATES resource_state,
                             double_ptr<Buffer> pp_buffer) {
  return CreateBuffer(size, heap_type, resource_state, D3D12_RESOURCE_FLAG_NONE, pp_buffer);
}

HRESULT Device::CreateBuffer(size_t size, D3D12_HEAP_TYPE heap_type, double_ptr<Buffer> pp_buffer) {
  return CreateBuffer(size, heap_type, D3D12_RESOURCE_STATE_COMMON, pp_buffer);
}

HRESULT Device::CreateBuffer(size_t size, double_ptr<Buffer> pp_buffer) {
  return CreateBuffer(size, D3D12_HEAP_TYPE_DEFAULT, pp_buffer);
}

HRESULT Device::CreateImage(const D3D12_RESOURCE_DESC &desc, double_ptr<Image> pp_image) {
  D3D12_CLEAR_VALUE clear_value = {};
  clear_value.Format = desc.Format;
  if (desc.Format == DXGI_FORMAT_D32_FLOAT) {
    clear_value.DepthStencil.Depth = 1.0f;
    clear_value.DepthStencil.Stencil = 0;
  } else {
    clear_value.Color[0] = 0.0f;
    clear_value.Color[1] = 0.0f;
    clear_value.Color[2] = 0.0f;
    clear_value.Color[3] = 1.0f;
  }
  const CD3DX12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE_DEFAULT);
  ComPtr<ID3D12Resource> image;
  RETURN_IF_FAILED_HR(
      device_->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                       &clear_value, IID_PPV_ARGS(&image)),
      "failed to create image.");

  pp_image.construct(image);

  return S_OK;
}

HRESULT Device::CreateImage(size_t width,
                            size_t height,
                            DXGI_FORMAT format,
                            D3D12_RESOURCE_FLAGS flags,
                            double_ptr<Image> pp_image) {
  CD3DX12_RESOURCE_DESC desc =
      CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, flags, D3D12_TEXTURE_LAYOUT_UNKNOWN, 0);
  return CreateImage(desc, pp_image);
}

HRESULT Device::CreateImage(size_t width, size_t height, DXGI_FORMAT format, double_ptr<Image> pp_image) {
  D3D12_RESOURCE_FLAGS flags;
  if (IsDepthFormat(format)) {
    flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
  } else {
    flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
  }
  return CreateImage(width, height, format, flags, pp_image);
}

HRESULT Device::CreateImageF32(size_t width, size_t height, double_ptr<Image> pp_image) {
  return CreateImage(width, height, DXGI_FORMAT_R32G32B32A32_FLOAT, pp_image);
}

HRESULT Device::CreateImageU8(size_t width, size_t height, double_ptr<Image> pp_image) {
  return CreateImage(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, pp_image);
}

HRESULT Device::CreateShaderModule(const void *compiled_shader_data,
                                   size_t size,
                                   double_ptr<ShaderModule> pp_shader_module) {
  std::vector<uint8_t> shader_code(static_cast<const uint8_t *>(compiled_shader_data),
                                   static_cast<const uint8_t *>(compiled_shader_data) + size);
  pp_shader_module.construct(shader_code);
  return S_OK;
}

HRESULT Device::CreateShaderModule(const ComPtr<ID3DBlob> &compiled_shader, double_ptr<ShaderModule> pp_shader_module) {
  return CreateShaderModule(compiled_shader->GetBufferPointer(), compiled_shader->GetBufferSize(), pp_shader_module);
}

HRESULT Device::CreateRootSignature(const CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC &desc,
                                    double_ptr<RootSignature> pp_root_signature) {
  D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

  // This is the highest version the sample supports. If CheckFeatureSupport
  // succeeds, the HighestVersion returned will not be greater than this.
  featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

  if (FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData)))) {
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
  }

  ComPtr<ID3DBlob> signature;
  ComPtr<ID3DBlob> error;

  auto hr = D3DX12SerializeVersionedRootSignature(&desc, featureData.HighestVersion, &signature, &error);
  if (FAILED(hr)) {
    if (error) {
      LogError("failed to serialize root signature: {}", static_cast<const char *>(error->GetBufferPointer()));
    }
    return hr;
  }

  ComPtr<ID3D12RootSignature> root_signature;

  RETURN_IF_FAILED_HR(device_->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                   IID_PPV_ARGS(&root_signature)),
                      "failed to create root signature.");

  pp_root_signature.construct(root_signature);

  return S_OK;
}

HRESULT Device::CreatePipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &desc,
                                    double_ptr<PipelineState> pp_pipeline_state) {
  ComPtr<ID3D12PipelineState> pipeline_state;

  RETURN_IF_FAILED_HR(device_->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline_state)),
                      "failed to create pipeline state.");

  pp_pipeline_state.construct(pipeline_state);

  return S_OK;
}

HRESULT Device::CreateBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS vertex_buffer,
                                                       D3D12_GPU_VIRTUAL_ADDRESS index_buffer,
                                                       uint32_t num_vertex,
                                                       uint32_t stride,
                                                       uint32_t primitive_count,
                                                       CommandQueue *queue,
                                                       CommandAllocator *allocator,
                                                       Fence *fence,
                                                       double_ptr<AccelerationStructure> pp_as) {
  D3D12_RAYTRACING_GEOMETRY_DESC geometry = {};
  geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
  geometry.Triangles.VertexBuffer.StartAddress = vertex_buffer;
  geometry.Triangles.VertexBuffer.StrideInBytes = stride;
  geometry.Triangles.VertexCount = num_vertex;
  geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
  geometry.Triangles.IndexBuffer = index_buffer;
  geometry.Triangles.IndexCount = primitive_count * 3;
  geometry.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
  geometry.Triangles.Transform3x4 = 0;
  geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS build_flags =
      D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS as_inputs = {};
  as_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
  as_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  as_inputs.pGeometryDescs = &geometry;
  as_inputs.NumDescs = 1;
  as_inputs.Flags = build_flags;

  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO as_prebuild_info = {};
  dxr_device_->GetRaytracingAccelerationStructurePrebuildInfo(&as_inputs, &as_prebuild_info);

  std::unique_ptr<Buffer> scratch_buffer;
  RETURN_IF_FAILED_HR(
      CreateBuffer(as_prebuild_info.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, &scratch_buffer),
      "failed to create scratch buffer.");
}

}  // namespace grassland::d3d12
