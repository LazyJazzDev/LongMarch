#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"

#include <cstring>

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalAccelerationStructure::MetalAccelerationStructure(MetalDevice *device) : device_(device) {
}

MetalDevice *MetalAccelerationStructure::Owner() const {
  return device_;
}

MTL::AccelerationStructure *MetalAccelerationStructure::NativeStructure() const {
  return structure_.get();
}

MTL::Buffer *MetalAccelerationStructure::CompactionQuery() const {
  return query_.get();
}

uint64_t MetalAccelerationStructure::ScratchSize(AccelerationStructureBuildMode mode) const {
  return mode == ACCELERATION_STRUCTURE_UPDATE ? update_scratch_ : build_scratch_;
}

uint64_t MetalAccelerationStructure::ReadCompactedSize() const {
  uint64_t size = 0;
  if (query_)
    std::memcpy(&size, query_->contents(), sizeof(size));
  return size;
}

bool MetalAccelerationStructure::AllowsUpdate() const {
  return allow_update_;
}

bool MetalAccelerationStructure::AllowsCompaction() const {
  return allow_compaction_;
}

bool MetalAccelerationStructure::IsCompactStorage() const {
  return compact_;
}

bool MetalAccelerationStructure::Allocate(MTL::AccelerationStructureDescriptor *desc,
                                          bool update,
                                          bool compaction,
                                          uint64_t compacted_size) {
  const auto sizes = device_->NativeDevice()->accelerationStructureSizes(desc);
  const auto size = compacted_size ? compacted_size : sizes.accelerationStructureSize;
  if (!size)
    return false;
  structure_ = NS::TransferPtr(device_->NativeDevice()->newAccelerationStructure(size));
  if (!structure_)
    return false;
  if (compaction) {
    query_ = NS::TransferPtr(device_->NativeDevice()->newBuffer(sizeof(uint64_t), MTL::ResourceStorageModeShared));
    if (!query_)
      return false;
    std::memset(query_->contents(), 0, sizeof(uint64_t));
  }
  build_scratch_ = sizes.buildScratchBufferSize;
  update_scratch_ = sizes.refitScratchBufferSize;
  allow_update_ = update;
  allow_compaction_ = compaction;
  compact_ = compacted_size != 0;
  return true;
}

MTL::PrimitiveAccelerationStructureDescriptor *NativeGeometries(
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers) {
  std::vector<const NS::Object *> geometries;
  for (size_t i = 0; i < desc.geometries.size(); ++i) {
    const auto &g = desc.geometries[i];
    auto native = MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor();
    native->setVertexFormat(MTL::AttributeFormatFloat3);
    native->setVertexStride(g.vertex_stride);
    if (g.index_count)
      native->setIndexType(g.index_format == INDEX_FORMAT_UINT16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32);
    native->setTriangleCount((g.index_count ? g.index_count : g.vertex_count) / 3);
    native->setOpaque(g.opaque);
    if (!buffers.empty()) {
      native->setVertexBuffer(static_cast<MetalBuffer *>(buffers[i].data.buffer)->NativeBuffer());
      native->setVertexBufferOffset(buffers[i].data.offset);
      if (g.index_count) {
        native->setIndexBuffer(static_cast<MetalBuffer *>(buffers[i].indices.buffer)->NativeBuffer());
        native->setIndexBufferOffset(buffers[i].indices.offset);
      }
    }
    geometries.push_back(native);
  }
  for (size_t i = 0; i < desc.aabbs.size(); ++i) {
    const auto &g = desc.aabbs[i];
    auto native = MTL::AccelerationStructureBoundingBoxGeometryDescriptor::descriptor();
    native->setBoundingBoxStride(g.stride);
    native->setBoundingBoxCount(g.count);
    native->setOpaque(g.opaque);
    if (!buffers.empty()) {
      native->setBoundingBoxBuffer(static_cast<MetalBuffer *>(buffers[i].data.buffer)->NativeBuffer());
      native->setBoundingBoxBufferOffset(buffers[i].data.offset);
    }
    geometries.push_back(native);
  }
  auto native = MTL::PrimitiveAccelerationStructureDescriptor::descriptor();
  native->setGeometryDescriptors(NS::Array::array(geometries.data(), geometries.size()));
  native->setUsage(desc.allow_update ? MTL::AccelerationStructureUsageRefit : MTL::AccelerationStructureUsageNone);
  return native;
}

MTL::InstanceAccelerationStructureDescriptor *NativeInstances(const TopLevelAccelerationStructureDesc &desc) {
  auto native = MTL::InstanceAccelerationStructureDescriptor::descriptor();
  native->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
  native->setInstanceDescriptorStride(sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor));
  native->setInstanceCount(desc.instance_count);
  native->setUsage(desc.allow_update ? MTL::AccelerationStructureUsageRefit : MTL::AccelerationStructureUsageNone);
  return native;
}

MetalBottomLevelAccelerationStructure::MetalBottomLevelAccelerationStructure(
    MetalDevice *device,
    const BottomLevelAccelerationStructureDesc &desc)
    : MetalAccelerationStructure(device),
      desc_(desc) {
}

MetalBottomLevelAccelerationStructure *MetalBottomLevelAccelerationStructure::Create(
    MetalDevice *device,
    const BottomLevelAccelerationStructureDesc &desc) {
  if (!device->SupportRayQuery() || !ValidateBottomLevelDesc(desc))
    return nullptr;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result =
      std::unique_ptr<MetalBottomLevelAccelerationStructure>(new MetalBottomLevelAccelerationStructure(device, desc));
  if (!result->Allocate(NativeGeometries(desc), desc.allow_update, desc.allow_compaction, desc.compacted_size))
    return nullptr;
  return result.release();
}

AccelerationStructureType MetalBottomLevelAccelerationStructure::GetType() const {
  return ACCELERATION_STRUCTURE_BOTTOM_LEVEL;
}

uint64_t MetalBottomLevelAccelerationStructure::GetCompactedSize() const {
  return ReadCompactedSize();
}

const BottomLevelAccelerationStructureDesc &MetalBottomLevelAccelerationStructure::GetDesc() const {
  return desc_;
}

MetalTopLevelAccelerationStructure::MetalTopLevelAccelerationStructure(MetalDevice *device,
                                                                       const TopLevelAccelerationStructureDesc &desc)
    : MetalAccelerationStructure(device),
      desc_(desc) {
}

MetalTopLevelAccelerationStructure *MetalTopLevelAccelerationStructure::Create(
    MetalDevice *device,
    const TopLevelAccelerationStructureDesc &desc) {
  if (!device->SupportRayQuery() || !desc.instance_count)
    return nullptr;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result =
      std::unique_ptr<MetalTopLevelAccelerationStructure>(new MetalTopLevelAccelerationStructure(device, desc));
  if (!result->Allocate(NativeInstances(desc), desc.allow_update, desc.allow_compaction, desc.compacted_size))
    return nullptr;
  return result.release();
}

AccelerationStructureType MetalTopLevelAccelerationStructure::GetType() const {
  return ACCELERATION_STRUCTURE_TOP_LEVEL;
}

uint64_t MetalTopLevelAccelerationStructure::GetCompactedSize() const {
  return ReadCompactedSize();
}

const TopLevelAccelerationStructureDesc &MetalTopLevelAccelerationStructure::GetDesc() const {
  return desc_;
}

const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &MetalTopLevelAccelerationStructure::BottomLevels()
    const {
  return bottom_levels_;
}

void MetalTopLevelAccelerationStructure::CommitDependencies(
    std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> refs) {
  bottom_levels_ = std::move(refs);
}
}  // namespace grassland::rhi::backend::metal
