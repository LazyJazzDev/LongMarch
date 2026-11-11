#pragma once
#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12AccelerationStructure, "c8f939f7-1266-44ee-ad3e-f0efb8e7007c")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12TopLevelAccelerationStructure,
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
  virtual uint64_t ReadCompactedSize() const = 0;
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
