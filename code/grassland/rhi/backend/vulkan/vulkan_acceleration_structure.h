#pragma once
#include "grassland/rhi/backend/backend_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanAccelerationStructure {
 public:
  VulkanDevice *Owner() const;
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
  explicit VulkanAccelerationStructure(VulkanDevice *device);
  ~VulkanAccelerationStructure();
  bool Allocate(const VkAccelerationStructureBuildGeometryInfoKHR &info,
                const std::vector<uint32_t> &counts,
                uint64_t compacted_size);

 private:
  VulkanDevice *device_;
  VkBuffer buffer_{};
  VmaAllocation allocation_{};
  VkAccelerationStructureKHR structure_{};
  VkQueryPool query_{};
  VkBuildAccelerationStructureFlagsKHR flags_{};
  uint64_t address_{};
  uint64_t build_scratch_{}, update_scratch_{};
  bool allow_update_{}, allow_compaction_{}, compact_{};
};

class VulkanBottomLevelAccelerationStructure final : public Object<IBottomLevelAccelerationStructure>,
                                                     public VulkanAccelerationStructure {
 public:
  static VulkanBottomLevelAccelerationStructure *Create(VulkanDevice *device,
                                                        const BottomLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const BottomLevelAccelerationStructureDesc &GetDesc() const override;

 private:
  VulkanBottomLevelAccelerationStructure(VulkanDevice *device, const BottomLevelAccelerationStructureDesc &desc);
  BottomLevelAccelerationStructureDesc desc_;
};

class VulkanTopLevelAccelerationStructure final : public Object<ITopLevelAccelerationStructure>,
                                                  public VulkanAccelerationStructure {
 public:
  static VulkanTopLevelAccelerationStructure *Create(VulkanDevice *device,
                                                     const TopLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const TopLevelAccelerationStructureDesc &GetDesc() const override;
  const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const;
  void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references);

 private:
  VulkanTopLevelAccelerationStructure(VulkanDevice *device, const TopLevelAccelerationStructureDesc &desc);
  TopLevelAccelerationStructureDesc desc_;
  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;
};

std::vector<VkAccelerationStructureGeometryKHR> NativeGeometries(
    VulkanDevice *device,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers = {});
}  // namespace grassland::rhi::backend::vulkan
