#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"

#include <cstring>

#include "grassland/rhi/backend/metal/metal_acceleration_structure_impl.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure_state.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
void IMetalAccelerationStructure::InitializeMetalAccelerationStructure(IMetalDevice *device) {
  StateMetalAccelerationStructure().device_ = device;
}

IMetalDevice *IMetalAccelerationStructure::Owner() const {
  return StateMetalAccelerationStructure().device_;
}

MTL::AccelerationStructure *IMetalAccelerationStructure::NativeStructure() const {
  return StateMetalAccelerationStructure().structure_.get();
}

MTL::Buffer *IMetalAccelerationStructure::CompactionQuery() const {
  return StateMetalAccelerationStructure().query_.get();
}

uint64_t IMetalAccelerationStructure::ScratchSize(AccelerationStructureBuildMode mode) const {
  return mode == ACCELERATION_STRUCTURE_UPDATE ? StateMetalAccelerationStructure().update_scratch_
                                               : StateMetalAccelerationStructure().build_scratch_;
}

uint64_t IMetalAccelerationStructure::ReadCompactedSize() const {
  uint64_t size = 0;
  if (StateMetalAccelerationStructure().query_)
    std::memcpy(&size, StateMetalAccelerationStructure().query_->contents(), sizeof(size));
  return size;
}

bool IMetalAccelerationStructure::AllowsUpdate() const {
  return StateMetalAccelerationStructure().allow_update_;
}

bool IMetalAccelerationStructure::AllowsCompaction() const {
  return StateMetalAccelerationStructure().allow_compaction_;
}

bool IMetalAccelerationStructure::IsCompactStorage() const {
  return StateMetalAccelerationStructure().compact_;
}

bool IMetalAccelerationStructure::Allocate(MTL::AccelerationStructureDescriptor *desc,
                                           bool update,
                                           bool compaction,
                                           uint64_t compacted_size) {
  const auto sizes = StateMetalAccelerationStructure().device_->NativeDevice()->accelerationStructureSizes(desc);
  const auto size = compacted_size ? compacted_size : sizes.accelerationStructureSize;
  if (!size)
    return false;
  StateMetalAccelerationStructure().structure_ =
      NS::TransferPtr(StateMetalAccelerationStructure().device_->NativeDevice()->newAccelerationStructure(size));
  if (!StateMetalAccelerationStructure().structure_)
    return false;
  if (compaction) {
    StateMetalAccelerationStructure().query_ =
        NS::TransferPtr(StateMetalAccelerationStructure().device_->NativeDevice()->newBuffer(
            sizeof(uint64_t), MTL::ResourceStorageModeShared));
    if (!StateMetalAccelerationStructure().query_)
      return false;
    std::memset(StateMetalAccelerationStructure().query_->contents(), 0, sizeof(uint64_t));
  }
  StateMetalAccelerationStructure().build_scratch_ = sizes.buildScratchBufferSize;
  StateMetalAccelerationStructure().update_scratch_ = sizes.refitScratchBufferSize;
  StateMetalAccelerationStructure().allow_update_ = update;
  StateMetalAccelerationStructure().allow_compaction_ = compaction;
  StateMetalAccelerationStructure().compact_ = compacted_size != 0;
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
      native->setVertexBuffer(rhi::QueryInterfacePtr<IMetalBuffer>(buffers[i].data.buffer).Get()->NativeBuffer());
      native->setVertexBufferOffset(buffers[i].data.offset);
      if (g.index_count) {
        native->setIndexBuffer(rhi::QueryInterfacePtr<IMetalBuffer>(buffers[i].indices.buffer).Get()->NativeBuffer());
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
      native->setBoundingBoxBuffer(rhi::QueryInterfacePtr<IMetalBuffer>(buffers[i].data.buffer).Get()->NativeBuffer());
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

MetalBottomLevelAccelerationStructureImpl::MetalBottomLevelAccelerationStructureImpl(
    IMetalDevice *device,
    const BottomLevelAccelerationStructureDesc &desc)
    : desc_(desc) {
  InitializeMetalAccelerationStructure(device);
}

IMetalBottomLevelAccelerationStructure *MetalBottomLevelAccelerationStructureImpl::Create(
    IMetalDevice *device,
    const BottomLevelAccelerationStructureDesc &desc) {
  if (!device->SupportRayQuery() || !ValidateBottomLevelDesc(desc))
    return nullptr;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result = std::unique_ptr<MetalBottomLevelAccelerationStructureImpl>(
      new MetalBottomLevelAccelerationStructureImpl(device, desc));
  if (!result->Allocate(NativeGeometries(desc), desc.allow_update, desc.allow_compaction, desc.compacted_size))
    return nullptr;
  return result.release();
}

AccelerationStructureType MetalBottomLevelAccelerationStructureImpl::GetType() const {
  return ACCELERATION_STRUCTURE_BOTTOM_LEVEL;
}

uint64_t MetalBottomLevelAccelerationStructureImpl::GetCompactedSize() const {
  return ReadCompactedSize();
}

const BottomLevelAccelerationStructureDesc &MetalBottomLevelAccelerationStructureImpl::GetDesc() const {
  return desc_;
}

MetalTopLevelAccelerationStructureImpl::MetalTopLevelAccelerationStructureImpl(
    IMetalDevice *device,
    const TopLevelAccelerationStructureDesc &desc)
    : desc_(desc) {
  InitializeMetalAccelerationStructure(device);
}

IMetalTopLevelAccelerationStructure *MetalTopLevelAccelerationStructureImpl::Create(
    IMetalDevice *device,
    const TopLevelAccelerationStructureDesc &desc) {
  if (!device->SupportRayQuery() || !desc.instance_count)
    return nullptr;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result =
      std::unique_ptr<MetalTopLevelAccelerationStructureImpl>(new MetalTopLevelAccelerationStructureImpl(device, desc));
  if (!result->Allocate(NativeInstances(desc), desc.allow_update, desc.allow_compaction, desc.compacted_size))
    return nullptr;
  return result.release();
}

AccelerationStructureType MetalTopLevelAccelerationStructureImpl::GetType() const {
  return ACCELERATION_STRUCTURE_TOP_LEVEL;
}

uint64_t MetalTopLevelAccelerationStructureImpl::GetCompactedSize() const {
  return ReadCompactedSize();
}

const TopLevelAccelerationStructureDesc &MetalTopLevelAccelerationStructureImpl::GetDesc() const {
  return desc_;
}

const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &MetalTopLevelAccelerationStructureImpl::BottomLevels()
    const {
  return bottom_levels_;
}

void MetalTopLevelAccelerationStructureImpl::CommitDependencies(
    std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> refs) {
  bottom_levels_ = std::move(refs);
}
}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalBottomLevelAccelerationStructure,
                   "5efa0e0c-89c2-4fdf-8295-1b1e502f6363")

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalTopLevelAccelerationStructure,
                   "240cc2a8-c334-4c86-b106-32c7fcc0f439")

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalAccelerationStructure, "15b08793-ad90-4431-b925-abab83480476")

namespace grassland::rhi::backend::metal {
MetalAccelerationStructureState::MetalAccelerationStructureState() = default;
MetalAccelerationStructureState::~MetalAccelerationStructureState() = default;
}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
IMetalBottomLevelAccelerationStructure *IMetalBottomLevelAccelerationStructure::Create(
    IMetalDevice *device,
    const BottomLevelAccelerationStructureDesc &desc) {
  return MetalBottomLevelAccelerationStructureImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
IMetalTopLevelAccelerationStructure *IMetalTopLevelAccelerationStructure::Create(
    IMetalDevice *device,
    const TopLevelAccelerationStructureDesc &desc) {
  return MetalTopLevelAccelerationStructureImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::metal
