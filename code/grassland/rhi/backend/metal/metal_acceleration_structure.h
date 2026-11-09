#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_acceleration_structure.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalAccelerationStructure {
 public:
  MetalDevice *Owner() const;
  MTL::AccelerationStructure *NativeStructure() const;
  MTL::Buffer *CompactionQuery() const;
  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const;
  uint64_t ReadCompactedSize() const;
  bool AllowsUpdate() const;
  bool AllowsCompaction() const;
  bool IsCompactStorage() const;

 protected:
  explicit MetalAccelerationStructure(MetalDevice *device);
  bool Allocate(MTL::AccelerationStructureDescriptor *descriptor,
                bool update,
                bool compaction,
                uint64_t compacted_size);

 private:
  MetalDevice *device_;
  NS::SharedPtr<MTL::AccelerationStructure> structure_;
  NS::SharedPtr<MTL::Buffer> query_;
  uint64_t build_scratch_{}, update_scratch_{};
  bool allow_update_{}, allow_compaction_{}, compact_{};
};

class MetalBottomLevelAccelerationStructure final : public Object<IBottomLevelAccelerationStructure>,
                                                    public MetalAccelerationStructure {
 public:
  static MetalBottomLevelAccelerationStructure *Create(MetalDevice *device,
                                                       const BottomLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const BottomLevelAccelerationStructureDesc &GetDesc() const override;

 private:
  MetalBottomLevelAccelerationStructure(MetalDevice *device, const BottomLevelAccelerationStructureDesc &desc);
  BottomLevelAccelerationStructureDesc desc_;
};

class MetalTopLevelAccelerationStructure final : public Object<ITopLevelAccelerationStructure>,
                                                 public MetalAccelerationStructure {
 public:
  static MetalTopLevelAccelerationStructure *Create(MetalDevice *device, const TopLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const TopLevelAccelerationStructureDesc &GetDesc() const override;
  const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const;
  void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references);

 private:
  MetalTopLevelAccelerationStructure(MetalDevice *device, const TopLevelAccelerationStructureDesc &desc);
  TopLevelAccelerationStructureDesc desc_;
  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;
};

MTL::PrimitiveAccelerationStructureDescriptor *NativeGeometries(
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
MTL::InstanceAccelerationStructureDescriptor *NativeInstances(const TopLevelAccelerationStructureDesc &desc);
}  // namespace grassland::rhi::backend::metal
