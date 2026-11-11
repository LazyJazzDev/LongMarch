#pragma once
#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanAccelerationStructure)
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanTopLevelAccelerationStructure)
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanBottomLevelAccelerationStructure)

namespace grassland::rhi::backend::vulkan {
struct VulkanAccelerationStructureState;

class IVulkanAccelerationStructure : public IObject {
 public:
  IVulkanDevice *Owner() const;
  VkAccelerationStructureKHR NativeStructure() const;
  VkQueryPool CompactionQuery() const;
  VkBuildAccelerationStructureFlagsKHR BuildFlags() const;
  uint64_t Address() const;
  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const;
  uint64_t ReadCompactedSize() const;
  bool AllowsUpdate() const;
  bool AllowsCompaction() const;
  bool IsCompactStorage() const;

 protected:
  void InitializeVulkanAccelerationStructure(IVulkanDevice *device);
  void DestroyVulkanAccelerationStructure();
  bool Allocate(const VkAccelerationStructureBuildGeometryInfoKHR &info,
                const std::vector<uint32_t> &counts,
                uint64_t compacted_size);
  virtual VulkanAccelerationStructureState &StateVulkanAccelerationStructure() = 0;
  virtual const VulkanAccelerationStructureState &StateVulkanAccelerationStructure() const = 0;
};

class IVulkanBottomLevelAccelerationStructure : public IBottomLevelAccelerationStructure,
                                                public IVulkanAccelerationStructure {
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
  static IVulkanBottomLevelAccelerationStructure *Create(IVulkanDevice *device,
                                                         const BottomLevelAccelerationStructureDesc &desc);
  virtual AccelerationStructureType GetType() const = 0;
  virtual uint64_t GetCompactedSize() const = 0;
  virtual const BottomLevelAccelerationStructureDesc &GetDesc() const = 0;
};

;

class IVulkanTopLevelAccelerationStructure : public ITopLevelAccelerationStructure,
                                             public IVulkanAccelerationStructure {
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
  static IVulkanTopLevelAccelerationStructure *Create(IVulkanDevice *device,
                                                      const TopLevelAccelerationStructureDesc &desc);
  virtual AccelerationStructureType GetType() const = 0;
  virtual uint64_t GetCompactedSize() const = 0;
  virtual const TopLevelAccelerationStructureDesc &GetDesc() const = 0;
  virtual const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const = 0;
  virtual void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) = 0;
};

;

std::vector<VkAccelerationStructureGeometryKHR> NativeGeometries(
    IVulkanDevice *device,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
}  // namespace grassland::rhi::backend::vulkan
