#pragma once
#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12AccelerationStructure {
 public:
  D3D12Device *Owner() const;
  ID3D12Resource *NativeResource() const;
  ID3D12Resource *CompactionReadback() const;
  ID3D12Resource *CompactionOutput() const;
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS NativeInputs() const;
  uint64_t Address() const;
  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const;
  uint64_t ReadCompactedSize() const;
  bool AllowsUpdate() const;
  bool AllowsCompaction() const;
  bool IsCompactStorage() const;

 protected:
  explicit D3D12AccelerationStructure(D3D12Device *device);
  ~D3D12AccelerationStructure();
  bool Allocate(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &inputs, uint64_t compacted_size);

 private:
  D3D12Device *device_;
  ComPtr<ID3D12Resource> resource_, query_buffer_, query_output_;
  ComPtr<D3D12MA::Allocation> allocation_, query_allocation_, query_output_allocation_;
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs_{};
  uint64_t build_scratch_{}, update_scratch_{};
  bool allow_update_{}, allow_compaction_{}, compact_{};
};

class D3D12BottomLevelAccelerationStructure final : public Object<IBottomLevelAccelerationStructure>,
                                                    public D3D12AccelerationStructure {
 public:
  static D3D12BottomLevelAccelerationStructure *Create(D3D12Device *device,
                                                       const BottomLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const BottomLevelAccelerationStructureDesc &GetDesc() const override;

 private:
  D3D12BottomLevelAccelerationStructure(D3D12Device *device, const BottomLevelAccelerationStructureDesc &desc);
  BottomLevelAccelerationStructureDesc desc_;
};

class D3D12TopLevelAccelerationStructure final : public Object<ITopLevelAccelerationStructure>,
                                                 public D3D12AccelerationStructure {
 public:
  static D3D12TopLevelAccelerationStructure *Create(D3D12Device *device, const TopLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const TopLevelAccelerationStructureDesc &GetDesc() const override;
  const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const;
  void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references);

 private:
  D3D12TopLevelAccelerationStructure(D3D12Device *device, const TopLevelAccelerationStructureDesc &desc);
  TopLevelAccelerationStructureDesc desc_;
  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;
};

std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> NativeGeometries(
    D3D12Device *device,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
}  // namespace grassland::rhi::backend::d3d12
