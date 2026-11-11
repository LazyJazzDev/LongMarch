#pragma once
#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalBottomLevelAccelerationStructureImpl final : public Object<IMetalBottomLevelAccelerationStructure> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IBottomLevelAccelerationStructure,
                            IMetalBottomLevelAccelerationStructure,
                            IBottomLevelAccelerationStructure,
                            IAccelerationStructure,
                            IMetalAccelerationStructure,
                            IObject)

  static IMetalBottomLevelAccelerationStructure *Create(IMetalDevice *device,
                                                        const BottomLevelAccelerationStructureDesc &desc);

  AccelerationStructureType GetType() const override;

  uint64_t GetCompactedSize() const override;

  const BottomLevelAccelerationStructureDesc &GetDesc() const override;

  ~MetalBottomLevelAccelerationStructureImpl() override = default;

  IMetalDevice *Owner() const override;

  MTL::AccelerationStructure *NativeStructure() const override;

  MTL::Buffer *CompactionQuery() const override;

  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const override;

  uint64_t ReadCompactedSize() const override;

  bool AllowsUpdate() const override;

  bool AllowsCompaction() const override;

  bool IsCompactStorage() const override;

 private:
  MetalBottomLevelAccelerationStructureImpl(IMetalDevice *device, const BottomLevelAccelerationStructureDesc &desc);

  BottomLevelAccelerationStructureDesc desc_;

  IMetalDevice *device_{};

  NS::SharedPtr<MTL::AccelerationStructure> structure_{};

  NS::SharedPtr<MTL::Buffer> query_{};

  uint64_t build_scratch_{}, update_scratch_{};

  bool allow_update_{}, allow_compaction_{}, compact_{};

  void InitializeMetalAccelerationStructure(IMetalDevice *device);

  bool Allocate(MTL::AccelerationStructureDescriptor *descriptor,
                bool update,
                bool compaction,
                uint64_t compacted_size);
};

class MetalTopLevelAccelerationStructureImpl final : public Object<IMetalTopLevelAccelerationStructure> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITopLevelAccelerationStructure,
                            IMetalTopLevelAccelerationStructure,
                            ITopLevelAccelerationStructure,
                            IAccelerationStructure,
                            IMetalAccelerationStructure,
                            IObject)

  static IMetalTopLevelAccelerationStructure *Create(IMetalDevice *device,
                                                     const TopLevelAccelerationStructureDesc &desc);

  AccelerationStructureType GetType() const override;

  uint64_t GetCompactedSize() const override;

  const TopLevelAccelerationStructureDesc &GetDesc() const override;

  const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const;

  void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references);

  ~MetalTopLevelAccelerationStructureImpl() override = default;

  IMetalDevice *Owner() const override;

  MTL::AccelerationStructure *NativeStructure() const override;

  MTL::Buffer *CompactionQuery() const override;

  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const override;

  uint64_t ReadCompactedSize() const override;

  bool AllowsUpdate() const override;

  bool AllowsCompaction() const override;

  bool IsCompactStorage() const override;

 private:
  MetalTopLevelAccelerationStructureImpl(IMetalDevice *device, const TopLevelAccelerationStructureDesc &desc);

  TopLevelAccelerationStructureDesc desc_;

  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;

  IMetalDevice *device_{};

  NS::SharedPtr<MTL::AccelerationStructure> structure_{};

  NS::SharedPtr<MTL::Buffer> query_{};

  uint64_t build_scratch_{}, update_scratch_{};

  bool allow_update_{}, allow_compaction_{}, compact_{};

  void InitializeMetalAccelerationStructure(IMetalDevice *device);

  bool Allocate(MTL::AccelerationStructureDescriptor *descriptor,
                bool update,
                bool compaction,
                uint64_t compacted_size);
};
}  // namespace grassland::rhi::backend::metal
