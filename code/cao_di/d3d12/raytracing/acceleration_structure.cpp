#include "cao_di/d3d12/raytracing/acceleration_structure.h"

#include "cao_di/d3d12/command_queue.h"

namespace CD::d3d12 {

AccelerationStructure::AccelerationStructure(Device *device, const ComPtr<ID3D12Resource> &as)
    : device_(device), as_(as) {
}

HRESULT AccelerationStructure::UpdateInstances(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC> &instances,
                                               CommandQueue *queue,
                                               Fence *fence,
                                               CommandAllocator *allocator) {
  ID3D12Device5 *device = device_->DXRDevice();
  RETURN_IF_FAILED_HR(as_->GetDevice(IID_PPV_ARGS(&device)), "failed to get DXR device.");

  ID3D12Resource *instance_buffer =
      device_->RequestInstanceBuffer(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * instances.size());
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
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE |
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO as_prebuild_info = {};
  device->GetRaytracingAccelerationStructurePrebuildInfo(&as_inputs, &as_prebuild_info);

  if (!as_ || as_->GetDesc().Width < as_prebuild_info.ResultDataMaxSizeInBytes) {
    RETURN_IF_FAILED_HR(
        d3d12::CreateBuffer(device_->Handle(), as_prebuild_info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, as_),
        "failed to create acceleration structure buffer.");
  }

  ID3D12Resource *scratch_buffer = device_->RequestScratchBuffer(as_prebuild_info.ScratchDataSizeInBytes);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC as_desc = {};
  as_desc.Inputs = as_inputs;
  as_desc.ScratchAccelerationStructureData = scratch_buffer->GetGPUVirtualAddress();
  as_desc.DestAccelerationStructureData = as_->GetGPUVirtualAddress();
  as_desc.SourceAccelerationStructureData = as_->GetGPUVirtualAddress();

  queue->SingleTimeCommand(fence, allocator, [&](ID3D12GraphicsCommandList *command_list) {
    ComPtr<ID3D12GraphicsCommandList4> command_list4;
    if (SUCCEEDED(command_list->QueryInterface(IID_PPV_ARGS(&command_list4)))) {
      command_list4->BuildRaytracingAccelerationStructure(&as_desc, 0, nullptr);
    }
  });

  return S_OK;
}

HRESULT AccelerationStructure::UpdateInstances(
    const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
    CommandQueue *queue,
    Fence *fence,
    CommandAllocator *allocator) {
  ID3D12Device5 *device = device_->DXRDevice();
  RETURN_IF_FAILED_HR(as_->GetDevice(IID_PPV_ARGS(&device)), "failed to get DXR device.");

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

  return UpdateInstances(instance_descs, queue, fence, allocator);
}

}  // namespace CD::d3d12
