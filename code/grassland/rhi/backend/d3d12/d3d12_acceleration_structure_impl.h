#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12BottomLevelAccelerationStructureImpl final
    : public Object<rhi::IBottomLevelAccelerationStructure, ID3D12AccelerationStructure> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IBottomLevelAccelerationStructure,
                            IBottomLevelAccelerationStructure,
                            IAccelerationStructure,
                            ID3D12AccelerationStructure)

  static IBottomLevelAccelerationStructure *Create(ID3D12Device *device,
                                                   const BottomLevelAccelerationStructureDesc &desc);

  AccelerationStructureType GetType() const override;

  uint64_t GetCompactedSize() const override;

  const BottomLevelAccelerationStructureDesc &GetDesc() const override;

  ~D3D12BottomLevelAccelerationStructureImpl() override;

  ID3D12Device *Owner() const override;

  ID3D12Resource *NativeResource() const override;

  ID3D12Resource *CompactionReadback() const override;

  ID3D12Resource *CompactionOutput() const override;

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS NativeInputs() const override;

  uint64_t Address() const override;

  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const override;

  uint64_t ReadCompactedSize() const override;

  bool AllowsUpdate() const override;

  bool AllowsCompaction() const override;

  bool IsCompactStorage() const override;

 private:
  D3D12BottomLevelAccelerationStructureImpl(ID3D12Device *device, const BottomLevelAccelerationStructureDesc &desc);

  BottomLevelAccelerationStructureDesc desc_;

  ID3D12Device *device_{};

  ComPtr<ID3D12Resource> resource_{}, query_buffer_{}, query_output_{};

  ComPtr<D3D12MA::Allocation> allocation_{}, query_allocation_{}, query_output_allocation_{};

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs_{};

  uint64_t build_scratch_{}, update_scratch_{};

  bool allow_update_{}, allow_compaction_{}, compact_{};

  void InitializeD3D12AccelerationStructure(ID3D12Device *device);

  void DestroyD3D12AccelerationStructure();

  bool Allocate(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &inputs, uint64_t compacted_size);
};

class D3D12TopLevelAccelerationStructureImpl final
    : public Object<rhi::ITopLevelAccelerationStructure, ID3D12TopLevelAccelerationStructure> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITopLevelAccelerationStructure,
                            ID3D12TopLevelAccelerationStructure,
                            ITopLevelAccelerationStructure,
                            IAccelerationStructure,
                            ID3D12AccelerationStructure)

  static ITopLevelAccelerationStructure *Create(ID3D12Device *device, const TopLevelAccelerationStructureDesc &desc);

  AccelerationStructureType GetType() const override;

  uint64_t GetCompactedSize() const override;

  const TopLevelAccelerationStructureDesc &GetDesc() const override;

  const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const;

  void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references);

  ~D3D12TopLevelAccelerationStructureImpl() override;

  ID3D12Device *Owner() const override;

  ID3D12Resource *NativeResource() const override;

  ID3D12Resource *CompactionReadback() const override;

  ID3D12Resource *CompactionOutput() const override;

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS NativeInputs() const override;

  uint64_t Address() const override;

  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const override;

  uint64_t ReadCompactedSize() const override;

  bool AllowsUpdate() const override;

  bool AllowsCompaction() const override;

  bool IsCompactStorage() const override;

 private:
  D3D12TopLevelAccelerationStructureImpl(ID3D12Device *device, const TopLevelAccelerationStructureDesc &desc);

  TopLevelAccelerationStructureDesc desc_;

  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;

  ID3D12Device *device_{};

  ComPtr<ID3D12Resource> resource_{}, query_buffer_{}, query_output_{};

  ComPtr<D3D12MA::Allocation> allocation_{}, query_allocation_{}, query_output_allocation_{};

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs_{};

  uint64_t build_scratch_{}, update_scratch_{};

  bool allow_update_{}, allow_compaction_{}, compact_{};

  void InitializeD3D12AccelerationStructure(ID3D12Device *device);

  void DestroyD3D12AccelerationStructure();

  bool Allocate(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &inputs, uint64_t compacted_size);
};
}  // namespace grassland::rhi::backend::d3d12
