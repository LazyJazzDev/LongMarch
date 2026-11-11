#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalAccelerationStructure)
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalTopLevelAccelerationStructure)
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalBottomLevelAccelerationStructure)
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_acceleration_structure.h"

namespace grassland::rhi::backend::metal {
struct MetalAccelerationStructureState;

class IMetalAccelerationStructure : public IObject {
 public:
  IMetalDevice *Owner() const;
  MTL::AccelerationStructure *NativeStructure() const;
  MTL::Buffer *CompactionQuery() const;
  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const;
  uint64_t ReadCompactedSize() const;
  bool AllowsUpdate() const;
  bool AllowsCompaction() const;
  bool IsCompactStorage() const;

 protected:
  void InitializeMetalAccelerationStructure(IMetalDevice *device);
  bool Allocate(MTL::AccelerationStructureDescriptor *descriptor,
                bool update,
                bool compaction,
                uint64_t compacted_size);
  virtual MetalAccelerationStructureState &StateMetalAccelerationStructure() = 0;
  virtual const MetalAccelerationStructureState &StateMetalAccelerationStructure() const = 0;
};

class IMetalBottomLevelAccelerationStructure : public IBottomLevelAccelerationStructure,
                                               public IMetalAccelerationStructure {
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
  static IMetalBottomLevelAccelerationStructure *Create(IMetalDevice *device,
                                                        const BottomLevelAccelerationStructureDesc &desc);
  virtual AccelerationStructureType GetType() const = 0;
  virtual uint64_t GetCompactedSize() const = 0;
  virtual const BottomLevelAccelerationStructureDesc &GetDesc() const = 0;
};

;

class IMetalTopLevelAccelerationStructure : public ITopLevelAccelerationStructure, public IMetalAccelerationStructure {
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
  static IMetalTopLevelAccelerationStructure *Create(IMetalDevice *device,
                                                     const TopLevelAccelerationStructureDesc &desc);
  virtual AccelerationStructureType GetType() const = 0;
  virtual uint64_t GetCompactedSize() const = 0;
  virtual const TopLevelAccelerationStructureDesc &GetDesc() const = 0;
  virtual const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const = 0;
  virtual void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) = 0;
};

;

MTL::PrimitiveAccelerationStructureDescriptor *NativeGeometries(
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
MTL::InstanceAccelerationStructureDescriptor *NativeInstances(const TopLevelAccelerationStructureDesc &desc);
}  // namespace grassland::rhi::backend::metal
