#include "grassland/d3d12/device.h"

#include <numeric>

#include "grassland/d3d12/buffer.h"
#include "grassland/d3d12/command_allocator.h"
#include "grassland/d3d12/command_list.h"
#include "grassland/d3d12/command_queue.h"
#include "grassland/d3d12/descriptor_heap.h"
#include "grassland/d3d12/fence.h"
#include "grassland/d3d12/image.h"
#include "grassland/d3d12/pipeline_state.h"
#include "grassland/d3d12/raytracing/raytracing.h"
#include "grassland/d3d12/root_signature.h"
#include "grassland/d3d12/shader_module.h"
#include "grassland/math/math_aabb.h"

namespace grassland::d3d12 {

Device::Device(const class Adapter &adapter, const D3D_FEATURE_LEVEL feature_level, ComPtr<ID3D12Device> device)
    : adapter_(adapter), feature_level_(feature_level), device_(std::move(device)) {
  // Get DXR interfaces
  ThrowIfFailed(device_->QueryInterface(IID_PPV_ARGS(&dxr_device_)), "failed to get DXR device interface.");
  device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &d3d12_options1_, sizeof(d3d12_options1_));
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

HRESULT Device::CreateFence(D3D12_FENCE_FLAGS fence_flags, double_ptr<Fence> pp_fence) {
  ComPtr<ID3D12Fence> fence;

  device_->CreateFence(0, fence_flags, IID_PPV_ARGS(&fence));

  pp_fence.construct(fence);

  return S_OK;
}

HRESULT Device::CreateFence(double_ptr<Fence> pp_fence) {
  return CreateFence(D3D12_FENCE_FLAG_NONE, pp_fence);
}

HRESULT Device::CreateBuffer(size_t size,
                             D3D12_HEAP_TYPE heap_type,
                             D3D12_HEAP_FLAGS heap_flags,
                             D3D12_RESOURCE_STATES resource_state,
                             D3D12_RESOURCE_FLAGS resource_flags,
                             double_ptr<Buffer> pp_buffer) {
  ComPtr<ID3D12Resource> buffer;

  RETURN_IF_FAILED_HR(::grassland::d3d12::CreateBuffer(device_.Get(), size, heap_type, heap_flags, resource_state,
                                                       resource_flags, buffer),
                      "failed to create buffer.");

  pp_buffer.construct(buffer, size);

  return S_OK;
}

HRESULT Device::CreateBuffer(size_t size,
                             D3D12_HEAP_TYPE heap_type,
                             D3D12_RESOURCE_STATES resource_state,
                             D3D12_RESOURCE_FLAGS resource_flags,
                             double_ptr<Buffer> pp_buffer) {
  return CreateBuffer(size, heap_type, D3D12_HEAP_FLAG_NONE, resource_state, resource_flags, pp_buffer);
}

HRESULT Device::CreateBuffer(size_t size,
                             D3D12_HEAP_TYPE heap_type,
                             D3D12_RESOURCE_STATES resource_state,
                             double_ptr<Buffer> pp_buffer) {
  return CreateBuffer(
      size, heap_type, resource_state,
      (heap_type == D3D12_HEAP_TYPE_DEFAULT) ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE,
      pp_buffer);
}

HRESULT Device::CreateBuffer(size_t size, D3D12_HEAP_TYPE heap_type, double_ptr<Buffer> pp_buffer) {
  return CreateBuffer(
      size, heap_type,
      (heap_type == D3D12_HEAP_TYPE_DEFAULT) ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON,
      pp_buffer);
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
  D3D12_RESOURCE_FLAGS flags{};
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
  CompiledShaderBlob shader_code;
  shader_code.data.resize(size);
  std::memcpy(shader_code.data.data(), compiled_shader_data, size);
  shader_code.entry_point = "main";
  return CreateShaderModule(shader_code, pp_shader_module);
}

HRESULT Device::CreateShaderModule(const CompiledShaderBlob &compiled_shader,
                                   double_ptr<ShaderModule> pp_shader_module) {
  pp_shader_module.construct(compiled_shader);
  return S_OK;
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

HRESULT Device::CreateBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS aabb_buffer,
                                                       uint32_t stride,
                                                       uint32_t num_aabb,
                                                       D3D12_RAYTRACING_GEOMETRY_FLAGS flags,
                                                       CommandQueue *queue,
                                                       Fence *fence,
                                                       CommandAllocator *allocator,
                                                       double_ptr<AccelerationStructure> pp_as) {
  D3D12_RAYTRACING_GEOMETRY_DESC geometry = {};
  geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
  geometry.AABBs.AABBs.StartAddress = aabb_buffer;
  geometry.AABBs.AABBs.StrideInBytes = stride;
  geometry.AABBs.AABBCount = num_aabb;
  geometry.Flags = flags;

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

  ID3D12Resource *scratch_buffer = RequestScratchBuffer(as_prebuild_info.ScratchDataSizeInBytes);

  ComPtr<ID3D12Resource> as;
  RETURN_IF_FAILED_HR(d3d12::CreateBuffer(Handle(), as_prebuild_info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, as),
                      "failed to create acceleration structure buffer.");

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC as_desc = {};
  as_desc.Inputs = as_inputs;
  as_desc.ScratchAccelerationStructureData = scratch_buffer->GetGPUVirtualAddress();
  as_desc.DestAccelerationStructureData = as->GetGPUVirtualAddress();
  as_desc.SourceAccelerationStructureData = 0;

  queue->SingleTimeCommand(fence, allocator, [&](ID3D12GraphicsCommandList *command_list) {
    ComPtr<ID3D12GraphicsCommandList4> command_list4;
    if (SUCCEEDED(command_list->QueryInterface(IID_PPV_ARGS(&command_list4)))) {
      command_list4->BuildRaytracingAccelerationStructure(&as_desc, 0, nullptr);
    }
  });

  pp_as.construct(this, as);
  return S_OK;
}

HRESULT Device::CreateBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS vertex_buffer,
                                                       D3D12_GPU_VIRTUAL_ADDRESS index_buffer,
                                                       uint32_t num_vertex,
                                                       uint32_t stride,
                                                       uint32_t primitive_count,
                                                       D3D12_RAYTRACING_GEOMETRY_FLAGS flags,
                                                       CommandQueue *queue,
                                                       Fence *fence,
                                                       CommandAllocator *allocator,
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
  geometry.Flags = flags;

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

  ID3D12Resource *scratch_buffer = RequestScratchBuffer(as_prebuild_info.ScratchDataSizeInBytes);

  ComPtr<ID3D12Resource> as;
  RETURN_IF_FAILED_HR(d3d12::CreateBuffer(Handle(), as_prebuild_info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, as),
                      "failed to create acceleration structure buffer.");

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC as_desc = {};
  as_desc.Inputs = as_inputs;
  as_desc.ScratchAccelerationStructureData = scratch_buffer->GetGPUVirtualAddress();
  as_desc.DestAccelerationStructureData = as->GetGPUVirtualAddress();
  as_desc.SourceAccelerationStructureData = 0;

  queue->SingleTimeCommand(fence, allocator, [&](ID3D12GraphicsCommandList *command_list) {
    ComPtr<ID3D12GraphicsCommandList4> command_list4;
    if (SUCCEEDED(command_list->QueryInterface(IID_PPV_ARGS(&command_list4)))) {
      command_list4->BuildRaytracingAccelerationStructure(&as_desc, 0, nullptr);
    }
  });

  pp_as.construct(this, as);
  return S_OK;
}

HRESULT Device::CreateBottomLevelAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS vertex_buffer,
                                                       D3D12_GPU_VIRTUAL_ADDRESS index_buffer,
                                                       uint32_t num_vertex,
                                                       uint32_t stride,
                                                       uint32_t primitive_count,
                                                       CommandQueue *queue,
                                                       Fence *fence,
                                                       CommandAllocator *allocator,
                                                       double_ptr<AccelerationStructure> pp_as) {
  return CreateBottomLevelAccelerationStructure(vertex_buffer, index_buffer, num_vertex, stride, primitive_count,
                                                D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION, queue,
                                                fence, allocator, pp_as);
}

HRESULT Device::CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                                       Buffer *index_buffer,
                                                       uint32_t stride,
                                                       CommandQueue *queue,
                                                       Fence *fence,
                                                       CommandAllocator *allocator,
                                                       double_ptr<AccelerationStructure> pp_as) {
  return CreateBottomLevelAccelerationStructure(
      vertex_buffer->Handle()->GetGPUVirtualAddress(), index_buffer->Handle()->GetGPUVirtualAddress(),
      vertex_buffer->Size() / stride, stride, index_buffer->Size() / (sizeof(uint32_t) * 3), queue, fence, allocator,
      pp_as);
}

HRESULT Device::CreateTopLevelAccelerationStructure(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC> &instances,
                                                    CommandQueue *queue,
                                                    Fence *fence,
                                                    CommandAllocator *allocator,
                                                    double_ptr<AccelerationStructure> pp_tlas) {
  ID3D12Resource *instance_buffer = RequestInstanceBuffer(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * instances.size());
  void *instance_buffer_ptr{};
  RETURN_IF_FAILED_HR(instance_buffer->Map(0, nullptr, &instance_buffer_ptr), "failed to map instance buffer.");
  std::memcpy(instance_buffer_ptr, instances.data(), instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
  instance_buffer->Unmap(0, nullptr);

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS as_inputs = {};
  as_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
  as_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  as_inputs.InstanceDescs = instance_buffer->GetGPUVirtualAddress();
  as_inputs.NumDescs = instances.size();
  as_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO as_prebuild_info = {};
  dxr_device_->GetRaytracingAccelerationStructurePrebuildInfo(&as_inputs, &as_prebuild_info);

  ID3D12Resource *scratch_buffer = RequestScratchBuffer(as_prebuild_info.ScratchDataSizeInBytes);

  ComPtr<ID3D12Resource> as;
  RETURN_IF_FAILED_HR(d3d12::CreateBuffer(Handle(), as_prebuild_info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, as),
                      "failed to create acceleration structure buffer.");

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC as_desc = {};
  as_desc.Inputs = as_inputs;
  as_desc.ScratchAccelerationStructureData = scratch_buffer->GetGPUVirtualAddress();
  as_desc.DestAccelerationStructureData = as->GetGPUVirtualAddress();
  as_desc.SourceAccelerationStructureData = 0;

  queue->SingleTimeCommand(fence, allocator, [&](ID3D12GraphicsCommandList *command_list) {
    ComPtr<ID3D12GraphicsCommandList4> command_list4;
    if (SUCCEEDED(command_list->QueryInterface(IID_PPV_ARGS(&command_list4)))) {
      command_list4->BuildRaytracingAccelerationStructure(&as_desc, 0, nullptr);
    }
  });

  pp_tlas.construct(this, as);
  return S_OK;
}

HRESULT Device::CreateTopLevelAccelerationStructure(
    const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
    CommandQueue *queue,
    Fence *fence,
    CommandAllocator *allocator,
    double_ptr<AccelerationStructure> pp_tlas) {
  std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instance_descs;
  instance_descs.reserve(objects.size());
  for (int i = 0; i < objects.size(); i++) {
    auto &object = objects[i];
    D3D12_RAYTRACING_INSTANCE_DESC instance_desc = {};
    instance_desc.Transform[0][0] = object.second[0][0];
    instance_desc.Transform[0][1] = object.second[1][0];
    instance_desc.Transform[0][2] = object.second[2][0];
    instance_desc.Transform[0][3] = object.second[3][0];
    instance_desc.Transform[1][0] = object.second[0][1];
    instance_desc.Transform[1][1] = object.second[1][1];
    instance_desc.Transform[1][2] = object.second[2][1];
    instance_desc.Transform[1][3] = object.second[3][1];
    instance_desc.Transform[2][0] = object.second[0][2];
    instance_desc.Transform[2][1] = object.second[1][2];
    instance_desc.Transform[2][2] = object.second[2][2];
    instance_desc.Transform[2][3] = object.second[3][2];
    instance_desc.InstanceID = i;
    instance_desc.InstanceMask = 0xFF;
    instance_desc.InstanceContributionToHitGroupIndex = 0;
    instance_desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
    instance_desc.AccelerationStructure = objects[i].first->Handle()->GetGPUVirtualAddress();
    instance_descs.push_back(instance_desc);
  }

  return CreateTopLevelAccelerationStructure(instance_descs, queue, fence, allocator, pp_tlas);
}

HRESULT Device::CreateRayTracingPipeline(RootSignature *root_signature,
                                         ShaderModule *ray_gen_shader,
                                         const std::vector<ShaderModule *> &miss_shaders,
                                         const std::vector<HitGroup> &hit_groups,
                                         const std::vector<ShaderModule *> &callable_shaders,
                                         double_ptr<RayTracingPipeline> pp_pipeline) {
  CD3DX12_STATE_OBJECT_DESC pipeline_desc(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);
  auto lib_ray_gen = pipeline_desc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
  auto ray_gen_code = ray_gen_shader->Handle();
  lib_ray_gen->SetDXILLibrary(&ray_gen_code);
  lib_ray_gen->DefineExport(L"RayGenMain", ray_gen_shader->EntryPoint().c_str());

  for (size_t i = 0; i < miss_shaders.size(); i++) {
    auto &miss_shader = miss_shaders[i];
    auto lib_miss = pipeline_desc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    auto miss_code = miss_shader->Handle();
    lib_miss->SetDXILLibrary(&miss_code);
    lib_miss->DefineExport((L"MissMain" + std::to_wstring(i)).c_str(), miss_shader->EntryPoint().c_str());
  }

  for (size_t i = 0; i < hit_groups.size(); i++) {
    auto &hit_group = hit_groups[i];
    auto obj_hit_group = pipeline_desc.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();

    auto lib_rchit = pipeline_desc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    auto rchit_code = hit_group.closest_hit_shader->Handle();
    lib_rchit->SetDXILLibrary(&rchit_code);
    lib_rchit->DefineExport((L"ClosestHitMain" + std::to_wstring(i)).c_str(),
                            hit_group.closest_hit_shader->EntryPoint().c_str());
    obj_hit_group->SetClosestHitShaderImport((L"ClosestHitMain" + std::to_wstring(i)).c_str());

    if (hit_group.intersection_shader) {
      auto lib_rint = pipeline_desc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
      auto rint_code = hit_group.intersection_shader->Handle();
      lib_rint->SetDXILLibrary(&rint_code);
      lib_rint->DefineExport((L"IntersectionMain" + std::to_wstring(i)).c_str(),
                             hit_group.intersection_shader->EntryPoint().c_str());
      obj_hit_group->SetIntersectionShaderImport((L"IntersectionMain" + std::to_wstring(i)).c_str());
    }

    if (hit_group.any_hit_shader) {
      auto lib_rahit = pipeline_desc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
      auto rahit_code = hit_group.any_hit_shader->Handle();
      lib_rahit->SetDXILLibrary(&rahit_code);
      lib_rahit->DefineExport((L"AnyHitMain" + std::to_wstring(i)).c_str(),
                              hit_group.any_hit_shader->EntryPoint().c_str());
      obj_hit_group->SetAnyHitShaderImport((L"AnyHitMain" + std::to_wstring(i)).c_str());
    }

    obj_hit_group->SetHitGroupExport((L"HitGroup" + std::to_wstring(i)).c_str());
    obj_hit_group->SetHitGroupType(hit_group.procedure ? D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE
                                                       : D3D12_HIT_GROUP_TYPE_TRIANGLES);
  }

  for (size_t i = 0; i < callable_shaders.size(); i++) {
    auto &callable_shader = callable_shaders[i];
    auto lib_callable = pipeline_desc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    auto callable_code = callable_shader->Handle();
    lib_callable->SetDXILLibrary(&callable_code);
    lib_callable->DefineExport((L"CallableMain" + std::to_wstring(i)).c_str(), callable_shader->EntryPoint().c_str());
  }

  auto shader_config = pipeline_desc.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
  shader_config->Config(512 * sizeof(float), 4 * sizeof(float));

  auto global_root_signature = pipeline_desc.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
  global_root_signature->SetRootSignature(root_signature->Handle());

  auto pipeline_config = pipeline_desc.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
  pipeline_config->Config(31);

  ComPtr<ID3D12StateObject> pipeline;

  RETURN_IF_FAILED_HR(dxr_device_->CreateStateObject(pipeline_desc, IID_PPV_ARGS(&pipeline)),
                      "failed to create ray tracing pipeline.");

  pp_pipeline.construct(pipeline, miss_shaders.size(), hit_groups.size(), callable_shaders.size());
  return S_OK;
}

HRESULT Device::CreateRayTracingPipeline(RootSignature *root_signature,
                                         ShaderModule *ray_gen_shader,
                                         ShaderModule *miss_shader,
                                         ShaderModule *closest_hit_shader,
                                         double_ptr<RayTracingPipeline> pp_pipeline) {
  return CreateRayTracingPipeline(root_signature, ray_gen_shader, {miss_shader},
                                  {{closest_hit_shader, nullptr, nullptr, false}}, {}, pp_pipeline);
}

HRESULT Device::CreateShaderTable(RayTracingPipeline *ray_tracing_pipeline,
                                  const std::vector<int32_t> &miss_shader_indices,
                                  const std::vector<int32_t> &hit_group_indices,
                                  const std::vector<int32_t> &callable_shader_indices,
                                  double_ptr<ShaderTable> pp_shader_table) const {
  ComPtr<ID3D12StateObjectProperties> pipeline_properties;
  RETURN_IF_FAILED_HR(ray_tracing_pipeline->Handle()->QueryInterface(IID_PPV_ARGS(&pipeline_properties)),
                      "failed to get pipeline properties.");

  const UINT shader_idenfitier_size = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
  const UINT shader_record_size = SizeAlignTo(shader_idenfitier_size, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
  const UINT shader_table_alignment = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;

  D3D12_GPU_VIRTUAL_ADDRESS accum_offset = 0;

  D3D12_GPU_VIRTUAL_ADDRESS ray_gen_shader_offset = 0;
  D3D12_GPU_VIRTUAL_ADDRESS miss_shader_offset = 0;
  D3D12_GPU_VIRTUAL_ADDRESS hit_group_offset = 0;
  D3D12_GPU_VIRTUAL_ADDRESS callable_shader_offset = 0;

  miss_shader_offset = accum_offset = SizeAlignTo(accum_offset + shader_record_size * 1, shader_table_alignment);

  hit_group_offset = accum_offset =
      SizeAlignTo(accum_offset + shader_record_size * miss_shader_indices.size(), shader_table_alignment);

  callable_shader_offset = accum_offset =
      SizeAlignTo(accum_offset + shader_record_size * hit_group_indices.size(), shader_table_alignment);

  accum_offset =
      SizeAlignTo(accum_offset + shader_record_size * callable_shader_indices.size(), shader_table_alignment);

  ComPtr<ID3D12Resource> buffer;
  RETURN_IF_FAILED_HR(d3d12::CreateBuffer(Handle(), accum_offset, D3D12_HEAP_TYPE_UPLOAD,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, buffer),
                      "failed to create shader binding table buffer.");
  uint8_t *data = nullptr;
  RETURN_IF_FAILED_HR(buffer->Map(0, nullptr, reinterpret_cast<void **>(&data)),
                      "failed to map shader binding table buffer.");

  void *shader_idenfitier = pipeline_properties->GetShaderIdentifier(L"RayGenMain");
  std::memcpy(data, shader_idenfitier, shader_idenfitier_size);

  for (size_t i = 0; i < miss_shader_indices.size(); i++) {
    auto miss_id = miss_shader_indices[i];
    shader_idenfitier = pipeline_properties->GetShaderIdentifier((L"MissMain" + std::to_wstring(miss_id)).c_str());
    std::memcpy(data + miss_shader_offset + i * shader_record_size, shader_idenfitier, shader_idenfitier_size);
  }

  for (size_t i = 0; i < hit_group_indices.size(); i++) {
    auto hit_group_id = hit_group_indices[i];
    shader_idenfitier = pipeline_properties->GetShaderIdentifier((L"HitGroup" + std::to_wstring(hit_group_id)).c_str());
    std::memcpy(data + hit_group_offset + i * shader_record_size, shader_idenfitier, shader_idenfitier_size);
  }

  for (size_t i = 0; i < callable_shader_indices.size(); i++) {
    auto callable_id = callable_shader_indices[i];
    shader_idenfitier =
        pipeline_properties->GetShaderIdentifier((L"CallableMain" + std::to_wstring(callable_id)).c_str());
    std::memcpy(data + callable_shader_offset + i * shader_record_size, shader_idenfitier, shader_idenfitier_size);
  }

  buffer->Unmap(0, nullptr);

  pp_shader_table.construct(buffer, ray_gen_shader_offset, miss_shader_offset, hit_group_offset, callable_shader_offset,
                            miss_shader_indices.size(), hit_group_indices.size(), callable_shader_indices.size());

  return S_OK;
}

HRESULT Device::CreateShaderTable(RayTracingPipeline *ray_tracing_pipeline,
                                  double_ptr<ShaderTable> pp_shader_table) const {
  std::vector<int32_t> miss_shader_indices(ray_tracing_pipeline->MissShaderCount());
  std::iota(miss_shader_indices.begin(), miss_shader_indices.end(), 0);
  std::vector<int32_t> hit_group_indices(ray_tracing_pipeline->HitGroupCount());
  std::iota(hit_group_indices.begin(), hit_group_indices.end(), 0);
  std::vector<int32_t> callable_shader_indices(ray_tracing_pipeline->CallableShaderCount());
  std::iota(callable_shader_indices.begin(), callable_shader_indices.end(), 0);
  return CreateShaderTable(ray_tracing_pipeline, miss_shader_indices, hit_group_indices, callable_shader_indices,
                           pp_shader_table);
}

ID3D12Resource *Device::RequestScratchBuffer(size_t size) {
  if (!scratch_buffer_ || scratch_buffer_->GetDesc().Width < size) {
    scratch_buffer_.Reset();
    d3d12::CreateBuffer(Handle(), size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, scratch_buffer_);
  }
  return scratch_buffer_.Get();
}

ID3D12Resource *Device::RequestInstanceBuffer(size_t size) {
  if (!instance_buffer_ || instance_buffer_->GetDesc().Width < size) {
    instance_buffer_.Reset();
    d3d12::CreateBuffer(Handle(), size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                        D3D12_RESOURCE_FLAG_NONE, instance_buffer_);
  }
  return instance_buffer_.Get();
}

}  // namespace grassland::d3d12
