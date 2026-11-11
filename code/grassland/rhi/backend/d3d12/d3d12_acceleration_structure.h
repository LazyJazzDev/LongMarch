#pragma once
#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12AccelerationStructure)
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12TopLevelAccelerationStructure)
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12BottomLevelAccelerationStructure)

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

class ID3D12BottomLevelAccelerationStructure : public IBottomLevelAccelerationStructure,
                                               public ID3D12AccelerationStructure {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

  static ID3D12BottomLevelAccelerationStructure *Create(ID3D12Device *device,
                                                        const BottomLevelAccelerationStructureDesc &desc);
  virtual AccelerationStructureType GetType() const = 0;
  virtual uint64_t GetCompactedSize() const = 0;
  virtual const BottomLevelAccelerationStructureDesc &GetDesc() const = 0;
};

class ID3D12TopLevelAccelerationStructure : public ITopLevelAccelerationStructure, public ID3D12AccelerationStructure {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

  static ID3D12TopLevelAccelerationStructure *Create(ID3D12Device *device,
                                                     const TopLevelAccelerationStructureDesc &desc);
  virtual AccelerationStructureType GetType() const = 0;
  virtual uint64_t GetCompactedSize() const = 0;
  virtual const TopLevelAccelerationStructureDesc &GetDesc() const = 0;
  virtual const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const = 0;
  virtual void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) = 0;
};

std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> NativeGeometries(
    ID3D12Device *device,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
}  // namespace grassland::rhi::backend::d3d12
