#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure_state.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12BottomLevelAccelerationStructureImpl final : public Object<ID3D12BottomLevelAccelerationStructure> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IBottomLevelAccelerationStructure,
                            ID3D12BottomLevelAccelerationStructure,
                            IBottomLevelAccelerationStructure,
                            IAccelerationStructure,
                            ID3D12AccelerationStructure,
                            IObject)
  static ID3D12BottomLevelAccelerationStructure *Create(ID3D12Device *device,
                                                        const BottomLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const BottomLevelAccelerationStructureDesc &GetDesc() const override;

 private:
  D3D12BottomLevelAccelerationStructureImpl(ID3D12Device *device, const BottomLevelAccelerationStructureDesc &desc);
  BottomLevelAccelerationStructureDesc desc_;

 public:
  ~D3D12BottomLevelAccelerationStructureImpl() override {
    DestroyD3D12AccelerationStructure();
  }

 private:
  D3D12AccelerationStructureState state_D3D12AccelerationStructure_;

  D3D12AccelerationStructureState &StateD3D12AccelerationStructure() override {
    return state_D3D12AccelerationStructure_;
  }

  const D3D12AccelerationStructureState &StateD3D12AccelerationStructure() const override {
    return state_D3D12AccelerationStructure_;
  }
};

class D3D12TopLevelAccelerationStructureImpl final : public Object<ID3D12TopLevelAccelerationStructure> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITopLevelAccelerationStructure,
                            ID3D12TopLevelAccelerationStructure,
                            ITopLevelAccelerationStructure,
                            IAccelerationStructure,
                            ID3D12AccelerationStructure,
                            IObject)
  static ID3D12TopLevelAccelerationStructure *Create(ID3D12Device *device,
                                                     const TopLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const TopLevelAccelerationStructureDesc &GetDesc() const override;
  const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const;
  void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references);

 private:
  D3D12TopLevelAccelerationStructureImpl(ID3D12Device *device, const TopLevelAccelerationStructureDesc &desc);
  TopLevelAccelerationStructureDesc desc_;
  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;

 public:
  ~D3D12TopLevelAccelerationStructureImpl() override {
    DestroyD3D12AccelerationStructure();
  }

 private:
  D3D12AccelerationStructureState state_D3D12AccelerationStructure_;

  D3D12AccelerationStructureState &StateD3D12AccelerationStructure() override {
    return state_D3D12AccelerationStructure_;
  }

  const D3D12AccelerationStructureState &StateD3D12AccelerationStructure() const override {
    return state_D3D12AccelerationStructure_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
