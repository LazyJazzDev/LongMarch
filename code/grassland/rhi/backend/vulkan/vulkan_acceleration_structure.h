#pragma once
#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanAccelerationStructure)
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanTopLevelAccelerationStructure)

namespace grassland::rhi::backend::vulkan {
class IVulkanAccelerationStructure : public IObject {
 public:
  virtual IVulkanDevice *Owner() const = 0;
  virtual VkAccelerationStructureKHR NativeStructure() const = 0;
  virtual VkQueryPool CompactionQuery() const = 0;
  virtual VkBuildAccelerationStructureFlagsKHR BuildFlags() const = 0;
  virtual uint64_t Address() const = 0;
  virtual uint64_t ScratchSize(AccelerationStructureBuildMode mode) const = 0;
  virtual uint64_t ReadCompactedSize() const = 0;
  virtual bool AllowsUpdate() const = 0;
  virtual bool AllowsCompaction() const = 0;
  virtual bool IsCompactStorage() const = 0;
};

IBottomLevelAccelerationStructure *CreateVulkanBottomLevelAccelerationStructure(
    IVulkanDevice *device,
    const BottomLevelAccelerationStructureDesc &desc);

class IVulkanTopLevelAccelerationStructure : public IVulkanAccelerationStructure {
 public:
  static ITopLevelAccelerationStructure *Create(IVulkanDevice *device, const TopLevelAccelerationStructureDesc &desc);
  virtual const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const = 0;
  virtual void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) = 0;
};

std::vector<VkAccelerationStructureGeometryKHR> NativeGeometries(
    IVulkanDevice *device,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
}  // namespace grassland::rhi::backend::vulkan
