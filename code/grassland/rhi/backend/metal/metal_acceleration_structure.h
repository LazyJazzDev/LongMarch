#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalAccelerationStructure,
                        grassland::rhi::IObject,
                        "15b08793-ad90-4431-b925-abab83480476")
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalTopLevelAccelerationStructure,
                        grassland::rhi::backend::metal::IMetalAccelerationStructure,
                        "240cc2a8-c334-4c86-b106-32c7fcc0f439")

namespace grassland::rhi::backend::metal {
class IMetalAccelerationStructure : public IObject {
 public:
  virtual IMetalDevice *Owner() const = 0;
  virtual MTL::AccelerationStructure *NativeStructure() const = 0;
  virtual MTL::Buffer *CompactionQuery() const = 0;
  virtual uint64_t ScratchSize(AccelerationStructureBuildMode mode) const = 0;
  virtual uint64_t ReadCompactedSize() const = 0;
  virtual bool AllowsUpdate() const = 0;
  virtual bool AllowsCompaction() const = 0;
  virtual bool IsCompactStorage() const = 0;
};

IBottomLevelAccelerationStructure *CreateMetalBottomLevelAccelerationStructure(
    IMetalDevice *device,
    const BottomLevelAccelerationStructureDesc &desc);

class IMetalTopLevelAccelerationStructure : public IMetalAccelerationStructure {
 public:
  static ITopLevelAccelerationStructure *Create(IMetalDevice *device, const TopLevelAccelerationStructureDesc &desc);
  virtual const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const = 0;
  virtual void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) = 0;
};

MTL::PrimitiveAccelerationStructureDescriptor *NativeGeometries(
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
MTL::InstanceAccelerationStructureDescriptor *NativeInstances(const TopLevelAccelerationStructureDesc &desc);
}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
class MetalBottomLevelAccelerationStructureImpl final
    : public Object<rhi::IBottomLevelAccelerationStructure, IMetalAccelerationStructure> {
 public:
  static IBottomLevelAccelerationStructure *Create(IMetalDevice *device,
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

class MetalTopLevelAccelerationStructureImpl final
    : public Object<rhi::ITopLevelAccelerationStructure, IMetalTopLevelAccelerationStructure> {
 public:
  static ITopLevelAccelerationStructure *Create(IMetalDevice *device, const TopLevelAccelerationStructureDesc &desc);

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
