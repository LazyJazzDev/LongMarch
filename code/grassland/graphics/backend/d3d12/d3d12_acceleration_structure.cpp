#include "grassland/graphics/backend/d3d12/d3d12_acceleration_structure.h"

#include "grassland/graphics/backend/d3d12/d3d12_core.h"

namespace grassland::graphics::backend {

D3D12AccelerationStructure::D3D12AccelerationStructure(D3D12Core *core,
                                                       Microsoft::WRL::ComPtr<ID3D12Resource> acceleration_structure,
                                                       int instance_count)
    : core_(core),
      acceleration_structure_(std::move(acceleration_structure)),
      instance_count_(instance_count) {
}

int D3D12AccelerationStructure::UpdateInstances(const std::vector<RayTracingInstance> &instances) {
  std::vector<D3D12_RAYTRACING_INSTANCE_DESC> d3d12_instances;
  d3d12_instances.reserve(instances.size());
  for (const auto &instance : instances) {
    d3d12_instances.emplace_back(RayTracingInstanceToD3D12RayTracingInstanceDesc(instance));
  }
  auto *device = core_->DXRDevice();
  auto *instance_buffer = core_->RequestInstanceBuffer(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * d3d12_instances.size());
  void *mapped = nullptr;
  d3d12::ThrowIfFailed(instance_buffer->Map(0, nullptr, &mapped), "Failed to map instance buffer");
  if (!d3d12_instances.empty()) {
    std::memcpy(mapped, d3d12_instances.data(), d3d12_instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
  }
  instance_buffer->Unmap(0, nullptr);

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
  inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
  inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  inputs.InstanceDescs = instance_buffer->GetGPUVirtualAddress();
  inputs.NumDescs = d3d12_instances.size();
  inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE |
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild{};
  device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild);
  bool rebuild = instance_count_ != d3d12_instances.size();
  if (!acceleration_structure_ || acceleration_structure_->GetDesc().Width < prebuild.ResultDataMaxSizeInBytes) {
    acceleration_structure_ = CreateNativeBuffer(
        core_->Device(), prebuild.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    rebuild = true;
  }
  if (rebuild) {
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                   D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
  }
  auto *scratch = core_->RequestScratchBuffer(prebuild.ScratchDataSizeInBytes);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
  build.Inputs = inputs;
  build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
  build.DestAccelerationStructureData = acceleration_structure_->GetGPUVirtualAddress();
  build.SourceAccelerationStructureData = rebuild ? 0 : acceleration_structure_->GetGPUVirtualAddress();
  d3d12::ThrowIfFailed(d3d12::SingleTimeCommand(core_->CommandQueue(), core_->SingleTimeCommandAllocator(),
                                                [&](ID3D12GraphicsCommandList *list) {
                                                  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> rt_list;
                                                  d3d12::ThrowIfFailed(list->QueryInterface(IID_PPV_ARGS(&rt_list)),
                                                                       "Failed to query DXR command list");
                                                  rt_list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
                                                }),
                       "Failed to update acceleration structure");
  instance_count_ = d3d12_instances.size();
  return 0;
}

}  // namespace grassland::graphics::backend
