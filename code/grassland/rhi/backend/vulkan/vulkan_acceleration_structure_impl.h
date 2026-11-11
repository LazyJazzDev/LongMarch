#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanBottomLevelAccelerationStructureImpl final
    : public Object<rhi::IBottomLevelAccelerationStructure, IVulkanAccelerationStructure> {
 public:
  static IBottomLevelAccelerationStructure *Create(IVulkanDevice *device,
                                                   const BottomLevelAccelerationStructureDesc &desc);

  AccelerationStructureType GetType() const override;

  uint64_t GetCompactedSize() const override;

  const BottomLevelAccelerationStructureDesc &GetDesc() const override;

  ~VulkanBottomLevelAccelerationStructureImpl() override;

  IVulkanDevice *Owner() const override;

  VkAccelerationStructureKHR NativeStructure() const override;

  VkQueryPool CompactionQuery() const override;

  VkBuildAccelerationStructureFlagsKHR BuildFlags() const override;

  uint64_t Address() const override;

  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const override;

  uint64_t ReadCompactedSize() const override;

  bool AllowsUpdate() const override;

  bool AllowsCompaction() const override;

  bool IsCompactStorage() const override;

 private:
  VulkanBottomLevelAccelerationStructureImpl(IVulkanDevice *device, const BottomLevelAccelerationStructureDesc &desc);

  BottomLevelAccelerationStructureDesc desc_;

  IVulkanDevice *device_{};

  VkBuffer buffer_{};

  VmaAllocation allocation_{};

  VkAccelerationStructureKHR structure_{};

  VkQueryPool query_{};

  VkBuildAccelerationStructureFlagsKHR flags_{};

  uint64_t address_{};

  uint64_t build_scratch_{}, update_scratch_{};

  bool allow_update_{}, allow_compaction_{}, compact_{};

  void InitializeVulkanAccelerationStructure(IVulkanDevice *device);

  void DestroyVulkanAccelerationStructure();

  bool Allocate(const VkAccelerationStructureBuildGeometryInfoKHR &info,
                const std::vector<uint32_t> &counts,
                uint64_t compacted_size);
};

class VulkanTopLevelAccelerationStructureImpl final
    : public Object<rhi::ITopLevelAccelerationStructure, IVulkanTopLevelAccelerationStructure> {
 public:
  static ITopLevelAccelerationStructure *Create(IVulkanDevice *device, const TopLevelAccelerationStructureDesc &desc);

  AccelerationStructureType GetType() const override;

  uint64_t GetCompactedSize() const override;

  const TopLevelAccelerationStructureDesc &GetDesc() const override;

  const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const;

  void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references);

  ~VulkanTopLevelAccelerationStructureImpl() override;

  IVulkanDevice *Owner() const override;

  VkAccelerationStructureKHR NativeStructure() const override;

  VkQueryPool CompactionQuery() const override;

  VkBuildAccelerationStructureFlagsKHR BuildFlags() const override;

  uint64_t Address() const override;

  uint64_t ScratchSize(AccelerationStructureBuildMode mode) const override;

  uint64_t ReadCompactedSize() const override;

  bool AllowsUpdate() const override;

  bool AllowsCompaction() const override;

  bool IsCompactStorage() const override;

 private:
  VulkanTopLevelAccelerationStructureImpl(IVulkanDevice *device, const TopLevelAccelerationStructureDesc &desc);

  TopLevelAccelerationStructureDesc desc_;

  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;

  IVulkanDevice *device_{};

  VkBuffer buffer_{};

  VmaAllocation allocation_{};

  VkAccelerationStructureKHR structure_{};

  VkQueryPool query_{};

  VkBuildAccelerationStructureFlagsKHR flags_{};

  uint64_t address_{};

  uint64_t build_scratch_{}, update_scratch_{};

  bool allow_update_{}, allow_compaction_{}, compact_{};

  void InitializeVulkanAccelerationStructure(IVulkanDevice *device);

  void DestroyVulkanAccelerationStructure();

  bool Allocate(const VkAccelerationStructureBuildGeometryInfoKHR &info,
                const std::vector<uint32_t> &counts,
                uint64_t compacted_size);
};
}  // namespace grassland::rhi::backend::vulkan
