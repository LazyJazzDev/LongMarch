#include <algorithm>
#include <cstring>

#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"

namespace grassland::rhi::backend::vulkan {
namespace {
uint64_t BufferAddress(VulkanDevice *device, VulkanBuffer *buffer) {
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  info.buffer = buffer->NativeBuffer();
  return vkGetBufferDeviceAddress(device->NativeDevice(), &info);
}

void AccelerationStructureBarrier(VkCommandBuffer command) {
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier,
                       0, nullptr, 0, nullptr);
}

uint64_t ScratchAlignment(VulkanDevice *device) {
  VkPhysicalDeviceAccelerationStructurePropertiesKHR as{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &as};
  vkGetPhysicalDeviceProperties2(device->NativePhysicalDevice(), &properties);
  return as.minAccelerationStructureScratchOffsetAlignment;
}
}  // namespace

bool VulkanCommandContext::CmdBuildBottomLevelAccelerationStructure(
    IBottomLevelAccelerationStructure *destination,
    const std::vector<AccelerationStructureGeometryBuffers> &geometries,
    AccelerationStructureBuildMode mode) {
  auto target = dynamic_cast<VulkanBottomLevelAccelerationStructure *>(destination);
  if (!target || target->Owner() != device_ || render_target_ ||
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
  auto alignment = ScratchAlignment(device_);
  auto scratch = device_->ScratchPool().Acquire(target->ScratchSize(mode) + alignment - 1);
  if (!scratch)
    return false;
  if (!RetainResource(destination))
    return false;
  if (!RetainResource(scratch.Get()))
    return false;
  if (!TransitionBuffer(scratch.Get(), VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR))
    return false;
  for (const auto &geometry : geometries) {
    for (const auto &view : {geometry.data, geometry.indices}) {
      if (!view.buffer)
        continue;
      auto buffer = static_cast<VulkanBuffer *>(view.buffer);
      if (!RetainResource(buffer))
        return false;
      if (!TransitionBuffer(buffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_SHADER_READ_BIT))
        return false;
    }
  }
  auto native = NativeGeometries(device_, target->GetDesc(), geometries);
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(native.size());
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (target->GetDesc().aabbs.empty()) {
      const auto &g = target->GetDesc().geometries[i];
      ranges[i].primitiveCount = (g.index_count ? g.index_count : g.vertex_count) / 3;
    } else
      ranges[i].primitiveCount = target->GetDesc().aabbs[i].count;
  }
  VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  build.flags = target->BuildFlags();
  build.mode = mode == ACCELERATION_STRUCTURE_UPDATE ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                                                     : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  build.srcAccelerationStructure = mode == ACCELERATION_STRUCTURE_UPDATE ? target->NativeStructure() : VK_NULL_HANDLE;
  build.dstAccelerationStructure = target->NativeStructure();
  build.geometryCount = uint32_t(native.size());
  build.pGeometries = native.data();
  build.scratchData.deviceAddress = (BufferAddress(device_, scratch.Get()) + alignment - 1) & ~(alignment - 1);
  auto range = ranges.data();
  auto command = device_->Procedures().CmdBuildAccelerationStructuresKHR;
  AccelerationStructureBarrier(command_);
  command(command_, 1, &build, &range);
  AccelerationStructureBarrier(command_);
  return true;
}

bool VulkanCommandContext::CmdBuildTopLevelAccelerationStructure(ITopLevelAccelerationStructure *destination,
                                                                 const std::vector<RayTracingInstance> &instances,
                                                                 AccelerationStructureBuildMode mode) {
  auto target = dynamic_cast<VulkanTopLevelAccelerationStructure *>(destination);
  if (!target || target->Owner() != device_ || render_target_ ||
      (mode != ACCELERATION_STRUCTURE_BUILD && mode != ACCELERATION_STRUCTURE_UPDATE) ||
      (mode == ACCELERATION_STRUCTURE_UPDATE && !target->AllowsUpdate()) ||
      (mode == ACCELERATION_STRUCTURE_BUILD && target->IsCompactStorage()) ||
      !ValidateInstances(target->GetDesc(), instances)) {
    recording_failed_ = true;
    return false;
  }
  std::vector<VkAccelerationStructureInstanceKHR> native(instances.size());
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &instance = instances[i];
    auto blas = dynamic_cast<VulkanBottomLevelAccelerationStructure *>(instance.blas);
    if (!blas || blas->Owner() != device_) {
      recording_failed_ = true;
      return false;
    }
    if (!RetainResource(blas))
      return false;
    std::memcpy(native[i].transform.matrix, instance.transform.data(), sizeof(native[i].transform.matrix));
    native[i].instanceCustomIndex = instance.custom_index;
    native[i].instanceShaderBindingTableRecordOffset = instance.hit_group_offset;
    native[i].mask = instance.mask;
    native[i].accelerationStructureReference = blas->Address();
    if (instance.disable_triangle_culling)
      native[i].flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    if (instance.front_counter_clockwise)
      native[i].flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
  }
  auto upload = device_->UploadPool().Acquire(
      std::max<uint64_t>(sizeof(VkAccelerationStructureInstanceKHR), native.size() * sizeof(native[0])));
  if (!upload)
    return false;
  if (!native.empty()) {
    if (!upload->Write(native.data(), native.size() * sizeof(native[0])))
      return false;
  }
  auto alignment = ScratchAlignment(device_);
  auto scratch = device_->ScratchPool().Acquire(target->ScratchSize(mode) + alignment - 1);
  if (!scratch)
    return false;
  if (!RetainResource(destination))
    return false;
  if (!RetainResource(upload.Get()))
    return false;
  if (!RetainResource(scratch.Get()))
    return false;
  if (!TransitionBuffer(upload.Get(), VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        VK_ACCESS_SHADER_READ_BIT))
    return false;
  if (!TransitionBuffer(scratch.Get(), VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR))
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
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  geometry.geometry.instances.data.deviceAddress = BufferAddress(device_, upload.Get());
  VkAccelerationStructureBuildRangeInfoKHR range{};
  range.primitiveCount = uint32_t(instances.size());
  auto ranges = &range;
  VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  build.flags = target->BuildFlags();
  build.mode = mode == ACCELERATION_STRUCTURE_UPDATE ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                                                     : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  build.srcAccelerationStructure = mode == ACCELERATION_STRUCTURE_UPDATE ? target->NativeStructure() : VK_NULL_HANDLE;
  build.dstAccelerationStructure = target->NativeStructure();
  build.geometryCount = 1;
  build.pGeometries = &geometry;
  build.scratchData.deviceAddress = (BufferAddress(device_, scratch.Get()) + alignment - 1) & ~(alignment - 1);
  auto command = device_->Procedures().CmdBuildAccelerationStructuresKHR;
  AccelerationStructureBarrier(command_);
  command(command_, 1, &build, &ranges);
  AccelerationStructureBarrier(command_);
  return true;
}

bool VulkanCommandContext::CmdQueryCompactedSize(IAccelerationStructure *structure) {
  auto target = dynamic_cast<VulkanAccelerationStructure *>(structure);
  if (!target || target->Owner() != device_ || !target->AllowsCompaction() || render_target_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(structure))
    return false;
  AccelerationStructureBarrier(command_);
  vkCmdResetQueryPool(command_, target->CompactionQuery(), 0, 1);
  auto native = target->NativeStructure();
  auto command = device_->Procedures().CmdWriteAccelerationStructuresPropertiesKHR;
  command(command_, 1, &native, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, target->CompactionQuery(), 0);
  return true;
}

bool VulkanCommandContext::CmdCompactAccelerationStructure(IAccelerationStructure *source,
                                                           IAccelerationStructure *destination) {
  auto from = dynamic_cast<VulkanAccelerationStructure *>(source);
  auto to = dynamic_cast<VulkanAccelerationStructure *>(destination);
  if (!from || !to || from == to || from->Owner() != device_ || to->Owner() != device_ || !from->AllowsCompaction() ||
      !to->IsCompactStorage() || source->GetType() != destination->GetType() || render_target_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(source))
    return false;
  if (!RetainResource(destination))
    return false;
  if (auto top = dynamic_cast<VulkanTopLevelAccelerationStructure *>(source)) {
    auto pending = as_dependencies_.find(top);
    auto dependencies = pending == as_dependencies_.end() ? top->BottomLevels() : pending->second;
    for (const auto &blas : dependencies)
      if (!RetainResource(blas.Get()))
        return false;
    auto target = static_cast<VulkanTopLevelAccelerationStructure *>(destination);
    for (const auto &blas : target->BottomLevels())
      if (!RetainResource(blas.Get()))
        return false;
    as_dependencies_[target] = std::move(dependencies);
  }
  VkCopyAccelerationStructureInfoKHR copy{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
  copy.src = from->NativeStructure();
  copy.dst = to->NativeStructure();
  copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
  auto command = device_->Procedures().CmdCopyAccelerationStructureKHR;
  AccelerationStructureBarrier(command_);
  command(command_, &copy);
  AccelerationStructureBarrier(command_);
  return true;
}
}  // namespace grassland::rhi::backend::vulkan
