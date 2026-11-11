#include <cmath>
#include <cstring>
#include <unordered_set>

#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_command_context_impl.h"

namespace grassland::rhi::backend::metal {
bool MetalCommandContextImpl::CmdBuildBottomLevelAccelerationStructure(
    IBottomLevelAccelerationStructure *destination,
    const std::vector<AccelerationStructureGeometryBuffers> &geometries,
    AccelerationStructureBuildMode mode) {
  auto target = rhi::QueryInterfacePtr<IMetalBottomLevelAccelerationStructure>(destination).Get();
  if (StateCommandContext().sealed_ || encoder_ || !target || target->Owner() != device_ ||
      (mode != ACCELERATION_STRUCTURE_BUILD && mode != ACCELERATION_STRUCTURE_UPDATE) ||
      (mode == ACCELERATION_STRUCTURE_UPDATE && !target->AllowsUpdate()) ||
      (mode == ACCELERATION_STRUCTURE_BUILD && target->IsCompactStorage()) ||
      !ValidateGeometryBuffers(target->GetDesc(), geometries, device_)) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto scratch = device_->ScratchPool().Acquire(target->ScratchSize(mode));
  if (!scratch || !RetainResource(destination) || !RetainResource(scratch.Get()))
    return false;
  for (const auto &g : geometries)
    for (const auto &v : {g.data, g.indices})
      if (v.buffer && !RetainResource(v.buffer))
        return false;
  auto descriptor = NativeGeometries(target->GetDesc(), geometries);
  auto encoder = commands_->accelerationStructureCommandEncoder();
  if (!encoder) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  // Buffers and AS use tracked storage. Serial encoders declare all scratch and
  // AS accesses so hazards are ordered on the device's single physical queue.
  encoder->useResource(scratch->NativeBuffer(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  if (mode == ACCELERATION_STRUCTURE_UPDATE)
    encoder->refitAccelerationStructure(target->NativeStructure(), descriptor, target->NativeStructure(),
                                        scratch->NativeBuffer(), 0);
  else
    encoder->buildAccelerationStructure(target->NativeStructure(), descriptor, scratch->NativeBuffer(), 0);
  encoder->endEncoding();
  return true;
}

bool MetalCommandContextImpl::CmdBuildTopLevelAccelerationStructure(ITopLevelAccelerationStructure *destination,
                                                                    const std::vector<RayTracingInstance> &instances,
                                                                    AccelerationStructureBuildMode mode) {
  auto target = rhi::QueryInterfacePtr<IMetalTopLevelAccelerationStructure>(destination).Get();
  if (StateCommandContext().sealed_ || encoder_ || !target || target->Owner() != device_ ||
      (mode != ACCELERATION_STRUCTURE_BUILD && mode != ACCELERATION_STRUCTURE_UPDATE) ||
      (mode == ACCELERATION_STRUCTURE_UPDATE && !target->AllowsUpdate()) ||
      (mode == ACCELERATION_STRUCTURE_BUILD && target->IsCompactStorage()) ||
      !ValidateInstances(target->GetDesc(), instances)) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  if (mode == ACCELERATION_STRUCTURE_UPDATE) {
    auto pending = as_dependencies_.find(target);
    auto count = pending == as_dependencies_.end() ? target->BottomLevels().size() : pending->second.size();
    if (count != instances.size()) {
      StateCommandContext().recording_failed_ = true;
      return false;
    }
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::vector<MTL::AccelerationStructureUserIDInstanceDescriptor> native(instances.size());
  std::vector<const NS::Object *> children;
  std::unordered_map<MTL::AccelerationStructure *, uint32_t> indices;
  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> dependencies;
  dependencies.reserve(instances.size());
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &instance = instances[i];
    auto blas = rhi::QueryInterfacePtr<IMetalBottomLevelAccelerationStructure>(instance.blas).Get();
    if (!blas || blas->Owner() != device_) {
      StateCommandContext().recording_failed_ = true;
      return false;
    }
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 3; ++r) {
        auto value = instance.transform[r * 4 + c];
        if (!std::isfinite(value)) {
          StateCommandContext().recording_failed_ = true;
          return false;
        }
        native[i].transformationMatrix.columns[c][r] = value;
      }
    if (!RetainResource(instance.blas))
      return false;
    dependencies.push_back(ObjectPtr<IBottomLevelAccelerationStructure>::Retain(instance.blas));
    auto [entry, inserted] = indices.emplace(blas->NativeStructure(), uint32_t(children.size()));
    if (inserted)
      children.push_back(blas->NativeStructure());
    native[i].accelerationStructureIndex = entry->second;
    native[i].userID = instance.custom_index;
    native[i].mask = instance.mask;
    // No RT pipeline/IFT is used; query instance IDs retain the public custom index.
    native[i].intersectionFunctionTableOffset = 0;
    native[i].options = MTL::AccelerationStructureInstanceOptionNone;
    if (instance.disable_triangle_culling)
      native[i].options |= MTL::AccelerationStructureInstanceOptionDisableTriangleCulling;
    if (instance.front_counter_clockwise)
      native[i].options |= MTL::AccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
  }
  auto upload = device_->UploadPool().Acquire(std::max<size_t>(sizeof(native[0]), native.size() * sizeof(native[0])));
  auto scratch = device_->ScratchPool().Acquire(target->ScratchSize(mode));
  if (!upload || !scratch)
    return false;
  if (!native.empty() && !upload->Write(native.data(), native.size() * sizeof(native[0])))
    return false;
  if (!RetainResource(destination) || !RetainResource(upload.Get()) || !RetainResource(scratch.Get()))
    return false;
  for (const auto &blas : target->BottomLevels())
    if (!RetainResource(blas.Get()))
      return false;
  auto descriptor = NativeInstances(target->GetDesc());
  descriptor->setInstanceCount(instances.size());
  descriptor->setInstanceDescriptorBuffer(upload->NativeBuffer());
  descriptor->setInstancedAccelerationStructures(NS::Array::array(children.data(), children.size()));
  // Publish only after submission, preserving dependencies if recording is abandoned.
  as_dependencies_[target] = std::move(dependencies);
  auto encoder = commands_->accelerationStructureCommandEncoder();
  if (!encoder) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  encoder->useResource(scratch->NativeBuffer(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  for (const auto &entry : indices)
    encoder->useResource(entry.first, MTL::ResourceUsageRead);
  if (mode == ACCELERATION_STRUCTURE_UPDATE)
    encoder->refitAccelerationStructure(target->NativeStructure(), descriptor, target->NativeStructure(),
                                        scratch->NativeBuffer(), 0);
  else
    encoder->buildAccelerationStructure(target->NativeStructure(), descriptor, scratch->NativeBuffer(), 0);
  encoder->endEncoding();
  return true;
}

bool MetalCommandContextImpl::CmdQueryCompactedSize(IAccelerationStructure *structure) {
  auto target = rhi::QueryInterfacePtr<IMetalAccelerationStructure>(structure).Get();
  if (StateCommandContext().sealed_ || encoder_ || !target || target->Owner() != device_ ||
      !target->AllowsCompaction()) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  if (!RetainResource(structure))
    return false;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto encoder = commands_->accelerationStructureCommandEncoder();
  if (!encoder) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  encoder->writeCompactedAccelerationStructureSize(target->NativeStructure(), target->CompactionQuery(), 0,
                                                   MTL::DataTypeULong);
  encoder->endEncoding();
  return true;
}

bool MetalCommandContextImpl::CmdCompactAccelerationStructure(IAccelerationStructure *source,
                                                              IAccelerationStructure *destination) {
  auto from = rhi::QueryInterfacePtr<IMetalAccelerationStructure>(source).Get();
  auto to = rhi::QueryInterfacePtr<IMetalAccelerationStructure>(destination).Get();
  if (StateCommandContext().sealed_ || encoder_ || !from || !to || from == to || from->Owner() != device_ ||
      to->Owner() != device_ || !from->AllowsCompaction() || !to->IsCompactStorage() ||
      source->GetType() != destination->GetType()) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  if (!RetainResource(source) || !RetainResource(destination))
    return false;
  if (auto top = rhi::QueryInterfacePtr<IMetalTopLevelAccelerationStructure>(source).Get()) {
    auto pending = as_dependencies_.find(top);
    auto dependencies = pending == as_dependencies_.end() ? top->BottomLevels() : pending->second;
    for (const auto &blas : dependencies)
      if (!RetainResource(blas.Get()))
        return false;
    auto target = rhi::QueryInterfacePtr<IMetalTopLevelAccelerationStructure>(destination).Get();
    for (const auto &blas : target->BottomLevels())
      if (!RetainResource(blas.Get()))
        return false;
    as_dependencies_[target] = std::move(dependencies);
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto encoder = commands_->accelerationStructureCommandEncoder();
  if (!encoder) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  encoder->copyAndCompactAccelerationStructure(from->NativeStructure(), to->NativeStructure());
  encoder->endEncoding();
  return true;
}

std::vector<MTL::Resource *> MetalCommandContextImpl::AccelerationResources() {
  std::vector<MTL::Resource *> resources;
  std::unordered_set<IBottomLevelAccelerationStructure *> seen;
  for (auto group : groups_) {
    if (!group)
      continue;
    for (size_t b = 0; b < group->GetLayout().bindings.size(); ++b) {
      if (group->GetLayout().bindings[b].type != BINDING_ACCELERATION_STRUCTURE)
        continue;
      for (const auto &ref : group->GetReferences()[b]) {
        auto top = rhi::QueryInterfacePtr<IMetalTopLevelAccelerationStructure>(ref.Get()).Get();
        auto pending = as_dependencies_.find(top);
        const auto &dependencies = pending == as_dependencies_.end() ? top->BottomLevels() : pending->second;
        for (const auto &blas : dependencies)
          if (seen.insert(blas.Get()).second) {
            RetainResource(blas.Get());
            resources.push_back(
                rhi::QueryInterfacePtr<IMetalBottomLevelAccelerationStructure>(blas.Get()).Get()->NativeStructure());
          }
      }
    }
  }
  return resources;
}
}  // namespace grassland::rhi::backend::metal
