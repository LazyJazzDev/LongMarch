#pragma once
#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12AccelerationStructure,
                        grassland::rhi::IObject,
                        "c8f939f7-1266-44ee-ad3e-f0efb8e7007c")
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12TopLevelAccelerationStructure,
                        grassland::rhi::backend::d3d12::ID3D12AccelerationStructure,
                        "4755c2d8-b4b6-477e-809f-939fc3683688")

namespace grassland::rhi::backend::d3d12 {
class ID3D12AccelerationStructure : public IObject {
 public:
  virtual ID3D12Device *Owner() const = 0;
  virtual ID3D12Resource *NativeResource() const = 0;
  virtual ID3D12Resource *CompactionReadback() const = 0;
  virtual ID3D12Resource *CompactionOutput() const = 0;
  virtual D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS NativeInputs() const = 0;
  virtual uint64_t Address() const = 0;
  virtual uint64_t ScratchSize(AccelerationStructureBuildMode mode) const = 0;
  virtual bool AllowsUpdate() const = 0;
  virtual bool AllowsCompaction() const = 0;
  virtual bool IsCompactStorage() const = 0;
};

IBottomLevelAccelerationStructure *CreateD3D12BottomLevelAccelerationStructure(
    ID3D12Device *device,
    const BottomLevelAccelerationStructureDesc &desc);

class ID3D12TopLevelAccelerationStructure : public ID3D12AccelerationStructure {
 public:
  static ITopLevelAccelerationStructure *Create(ID3D12Device *device, const TopLevelAccelerationStructureDesc &desc);
  virtual const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const = 0;
  virtual void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) = 0;
};

std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> NativeGeometries(
    ID3D12Device *device,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
class D3D12BottomLevelAccelerationStructureImpl final
    : public Object<rhi::IBottomLevelAccelerationStructure, ID3D12AccelerationStructure> {
 public:
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

  bool AllowsUpdate() const override;

  bool AllowsCompaction() const override;

  bool IsCompactStorage() const override;

 private:
  uint64_t ReadCompactedSize() const;

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

  bool AllowsUpdate() const override;

  bool AllowsCompaction() const override;

  bool IsCompactStorage() const override;

 private:
  uint64_t ReadCompactedSize() const;

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
