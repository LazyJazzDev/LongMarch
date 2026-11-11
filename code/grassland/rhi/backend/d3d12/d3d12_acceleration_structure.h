#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12AccelerationStructure;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::D3D12AccelerationStructure)
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12TopLevelAccelerationStructure;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12TopLevelAccelerationStructure)
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12BottomLevelAccelerationStructure;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12BottomLevelAccelerationStructure)
#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12Device;

struct D3D12AccelerationStructureState;

class D3D12AccelerationStructure : public IObject {
 public:
  ID3D12Device *Owner() const;
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
  void InitializeD3D12AccelerationStructure(ID3D12Device *device);
  void DestroyD3D12AccelerationStructure();
  bool Allocate(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &inputs, uint64_t compacted_size);
  virtual D3D12AccelerationStructureState &StateD3D12AccelerationStructure() = 0;
  virtual const D3D12AccelerationStructureState &StateD3D12AccelerationStructure() const = 0;
};

class ID3D12BottomLevelAccelerationStructure : public IBottomLevelAccelerationStructure,
                                               public D3D12AccelerationStructure {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

 public:
  static ID3D12BottomLevelAccelerationStructure *Create(ID3D12Device *device,
                                                        const BottomLevelAccelerationStructureDesc &desc);
  virtual AccelerationStructureType GetType() const = 0;
  virtual uint64_t GetCompactedSize() const = 0;
  virtual const BottomLevelAccelerationStructureDesc &GetDesc() const = 0;
};

;

class ID3D12TopLevelAccelerationStructure : public ITopLevelAccelerationStructure, public D3D12AccelerationStructure {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

 public:
  static ID3D12TopLevelAccelerationStructure *Create(ID3D12Device *device,
                                                     const TopLevelAccelerationStructureDesc &desc);
  virtual AccelerationStructureType GetType() const = 0;
  virtual uint64_t GetCompactedSize() const = 0;
  virtual const TopLevelAccelerationStructureDesc &GetDesc() const = 0;
  virtual const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const = 0;
  virtual void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) = 0;
};

;

std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> NativeGeometries(
    ID3D12Device *device,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
}  // namespace grassland::rhi::backend::d3d12
