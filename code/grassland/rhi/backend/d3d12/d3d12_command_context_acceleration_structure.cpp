#include <algorithm>
#include <cstring>

#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

namespace grassland::rhi::backend::d3d12 {
bool D3D12CommandContext::CmdBuildBottomLevelAccelerationStructure(
    IBottomLevelAccelerationStructure *destination,
    const std::vector<AccelerationStructureGeometryBuffers> &geometries,
    AccelerationStructureBuildMode mode) {
  auto target = dynamic_cast<D3D12BottomLevelAccelerationStructure *>(destination);
  if (!ray_list_ || !target || target->Owner() != device_ || rendering_active_ ||
      (mode != ACCELERATION_STRUCTURE_BUILD && mode != ACCELERATION_STRUCTURE_UPDATE) ||
      (mode == ACCELERATION_STRUCTURE_UPDATE && !target->AllowsUpdate()) ||
      (mode == ACCELERATION_STRUCTURE_BUILD && target->IsCompactStorage())) {
    recording_failed_ = true;
    return false;
  }
#if !defined(NDEBUG)
  if (!ValidateGeometryBuffers(target->GetDesc(), geometries, device_)) {
    recording_failed_ = true;
    return false;
  }
#endif
  auto scratch = device_->ScratchPool().Acquire(target->ScratchSize(mode));
  if (!scratch)
    return false;
  if (!RetainResource(destination))
    return false;
  if (!RetainResource(scratch.Get()))
    return false;
  if (!TransitionResource(scratch->NativeBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, scratch->DefaultState()))
    return false;
  for (const auto &geometry : geometries) {
    for (const auto &view : {geometry.data, geometry.indices}) {
      if (!view.buffer)
        continue;
      auto buffer = static_cast<D3D12Buffer *>(view.buffer);
      if (!RetainResource(buffer))
        return false;
      if (!TransitionResource(buffer->NativeBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                              buffer->DefaultState()))
        return false;
    }
  }
  auto native = NativeGeometries(device_, target->GetDesc(), geometries);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
  build.Inputs = target->NativeInputs();
  build.Inputs.pGeometryDescs = native.data();
  build.DestAccelerationStructureData = target->Address();
  build.ScratchAccelerationStructureData = scratch->NativeBuffer()->GetGPUVirtualAddress();
  if (mode == ACCELERATION_STRUCTURE_UPDATE) {
    build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    build.SourceAccelerationStructureData = target->Address();
  }
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.UAV.pResource = target->NativeResource();
  list_->ResourceBarrier(1, &barrier);
  ray_list_->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
  list_->ResourceBarrier(1, &barrier);
  return true;
}

bool D3D12CommandContext::CmdBuildTopLevelAccelerationStructure(ITopLevelAccelerationStructure *destination,
                                                                const std::vector<RayTracingInstance> &instances,
                                                                AccelerationStructureBuildMode mode) {
  auto target = dynamic_cast<D3D12TopLevelAccelerationStructure *>(destination);
  if (!ray_list_ || !target || target->Owner() != device_ || rendering_active_ ||
      (mode != ACCELERATION_STRUCTURE_BUILD && mode != ACCELERATION_STRUCTURE_UPDATE) ||
      (mode == ACCELERATION_STRUCTURE_UPDATE && !target->AllowsUpdate()) ||
      (mode == ACCELERATION_STRUCTURE_BUILD && target->IsCompactStorage()) ||
      !ValidateInstances(target->GetDesc(), instances)) {
    recording_failed_ = true;
    return false;
  }
  if (mode == ACCELERATION_STRUCTURE_UPDATE) {
    auto pending = as_dependencies_.find(target);
    const size_t previous_count =
        pending == as_dependencies_.end() ? target->BottomLevels().size() : pending->second.size();
    if (instances.size() != previous_count) {
      recording_failed_ = true;
      return false;
    }
  }
  std::vector<D3D12_RAYTRACING_INSTANCE_DESC> native(instances.size());
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &instance = instances[i];
#if !defined(NDEBUG)
    auto blas = dynamic_cast<D3D12BottomLevelAccelerationStructure *>(instance.blas);
    if (!blas || blas->Owner() != device_) {
      recording_failed_ = true;
      return false;
    }
#else
    auto blas = static_cast<D3D12BottomLevelAccelerationStructure *>(instance.blas);
#endif
    if (!RetainResource(blas))
      return false;
    std::memcpy(native[i].Transform, instance.transform.data(), sizeof(native[i].Transform));
    native[i].InstanceID = instance.custom_index;
    native[i].InstanceContributionToHitGroupIndex = instance.hit_group_offset;
    native[i].InstanceMask = instance.mask;
    native[i].AccelerationStructure = blas->Address();
    if (instance.disable_triangle_culling)
      native[i].Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
    if (instance.front_counter_clockwise)
      native[i].Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE;
  }
  auto upload = device_->UploadPool().Acquire(
      std::max<uint64_t>(sizeof(D3D12_RAYTRACING_INSTANCE_DESC), native.size() * sizeof(native[0])));
  if (!upload)
    return false;
  if (!native.empty()) {
    if (!upload->Write(native.data(), native.size() * sizeof(native[0])))
      return false;
  }
  auto scratch = device_->ScratchPool().Acquire(target->ScratchSize(mode));
  if (!scratch)
    return false;
  if (!RetainResource(destination))
    return false;
  if (!RetainResource(upload.Get()))
    return false;
  if (!RetainResource(scratch.Get()))
    return false;
  if (!TransitionResource(scratch->NativeBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, scratch->DefaultState()))
    return false;
  // Do not change an existing TLAS's ownership when a recording is abandoned.
  for (const auto &blas : target->BottomLevels())
    if (!RetainResource(blas.Get()))
      return false;
  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> dependencies;
  dependencies.reserve(instances.size());
  for (const auto &instance : instances)
    dependencies.push_back(ObjectPtr<IBottomLevelAccelerationStructure>::Retain(instance.blas));
  as_dependencies_[target] = std::move(dependencies);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
  build.Inputs = target->NativeInputs();
  build.Inputs.NumDescs = uint32_t(instances.size());
  build.Inputs.InstanceDescs = upload->NativeBuffer()->GetGPUVirtualAddress();
  build.DestAccelerationStructureData = target->Address();
  build.ScratchAccelerationStructureData = scratch->NativeBuffer()->GetGPUVirtualAddress();
  if (mode == ACCELERATION_STRUCTURE_UPDATE) {
    build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    build.SourceAccelerationStructureData = target->Address();
  }
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.UAV.pResource = target->NativeResource();
  list_->ResourceBarrier(1, &barrier);
  ray_list_->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
  list_->ResourceBarrier(1, &barrier);
  return true;
}

bool D3D12CommandContext::CmdQueryCompactedSize(IAccelerationStructure *structure) {
  auto target = dynamic_cast<D3D12AccelerationStructure *>(structure);
  if (!ray_list_ || !target || target->Owner() != device_ || !target->AllowsCompaction() || rendering_active_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(structure))
    return false;
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC query{};
  query.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
  query.DestBuffer = target->CompactionOutput()->GetGPUVirtualAddress();
  auto address = target->Address();
  ray_list_->EmitRaytracingAccelerationStructurePostbuildInfo(&query, 1, &address);
  if (!TransitionResource(target->CompactionOutput(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
    return false;
  list_->CopyBufferRegion(target->CompactionReadback(), 0, target->CompactionOutput(), 0, sizeof(uint64_t));
  return TransitionResource(target->CompactionOutput(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

bool D3D12CommandContext::CmdCompactAccelerationStructure(IAccelerationStructure *source,
                                                          IAccelerationStructure *destination) {
  auto from = dynamic_cast<D3D12AccelerationStructure *>(source);
  auto to = dynamic_cast<D3D12AccelerationStructure *>(destination);
  if (!ray_list_ || !from || !to || from == to || from->Owner() != device_ || to->Owner() != device_ ||
      !from->AllowsCompaction() || !to->IsCompactStorage() || source->GetType() != destination->GetType() ||
      rendering_active_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(source))
    return false;
  if (!RetainResource(destination))
    return false;
  if (auto top = dynamic_cast<D3D12TopLevelAccelerationStructure *>(source)) {
    auto pending = as_dependencies_.find(top);
    auto dependencies = pending == as_dependencies_.end() ? top->BottomLevels() : pending->second;
    for (const auto &blas : dependencies)
      if (!RetainResource(blas.Get()))
        return false;
    auto target = static_cast<D3D12TopLevelAccelerationStructure *>(destination);
    for (const auto &blas : target->BottomLevels())
      if (!RetainResource(blas.Get()))
        return false;
    as_dependencies_[target] = std::move(dependencies);
  }
  ray_list_->CopyRaytracingAccelerationStructure(to->Address(), from->Address(),
                                                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.UAV.pResource = to->NativeResource();
  list_->ResourceBarrier(1, &barrier);
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
