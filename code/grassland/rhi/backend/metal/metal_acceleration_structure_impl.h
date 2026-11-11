#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure_state.h"

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

 private:
  MetalBottomLevelAccelerationStructureImpl(IMetalDevice *device, const BottomLevelAccelerationStructureDesc &desc);
  BottomLevelAccelerationStructureDesc desc_;

 public:
  ~MetalBottomLevelAccelerationStructureImpl() override {
  }

 private:
  MetalAccelerationStructureState state_MetalAccelerationStructure_;

  MetalAccelerationStructureState &StateMetalAccelerationStructure() override {
    return state_MetalAccelerationStructure_;
  }

  const MetalAccelerationStructureState &StateMetalAccelerationStructure() const override {
    return state_MetalAccelerationStructure_;
  }
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

 private:
  MetalTopLevelAccelerationStructureImpl(IMetalDevice *device, const TopLevelAccelerationStructureDesc &desc);
  TopLevelAccelerationStructureDesc desc_;
  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;

 public:
  ~MetalTopLevelAccelerationStructureImpl() override {
  }

 private:
  MetalAccelerationStructureState state_MetalAccelerationStructure_;

  MetalAccelerationStructureState &StateMetalAccelerationStructure() override {
    return state_MetalAccelerationStructure_;
  }

  const MetalAccelerationStructureState &StateMetalAccelerationStructure() const override {
    return state_MetalAccelerationStructure_;
  }
};
}  // namespace grassland::rhi::backend::metal
