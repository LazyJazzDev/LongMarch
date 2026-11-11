#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalAccelerationStructure)
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalTopLevelAccelerationStructure)

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
