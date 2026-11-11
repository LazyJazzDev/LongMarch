#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure_state.h"

namespace grassland::rhi::backend::vulkan {
class VulkanBottomLevelAccelerationStructureImpl final : public Object<IVulkanBottomLevelAccelerationStructure> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IBottomLevelAccelerationStructure,
                            IVulkanBottomLevelAccelerationStructure,
                            IBottomLevelAccelerationStructure,
                            IAccelerationStructure,
                            VulkanAccelerationStructure,
                            IObject)
  static IVulkanBottomLevelAccelerationStructure *Create(IVulkanDevice *device,
                                                         const BottomLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const BottomLevelAccelerationStructureDesc &GetDesc() const override;

 private:
  VulkanBottomLevelAccelerationStructureImpl(IVulkanDevice *device, const BottomLevelAccelerationStructureDesc &desc);
  BottomLevelAccelerationStructureDesc desc_;

 public:
  ~VulkanBottomLevelAccelerationStructureImpl() override {
    DestroyVulkanAccelerationStructure();
  }

 private:
  VulkanAccelerationStructureState state_VulkanAccelerationStructure_;

  VulkanAccelerationStructureState &StateVulkanAccelerationStructure() override {
    return state_VulkanAccelerationStructure_;
  }

  const VulkanAccelerationStructureState &StateVulkanAccelerationStructure() const override {
    return state_VulkanAccelerationStructure_;
  }
};

class VulkanTopLevelAccelerationStructureImpl final : public Object<IVulkanTopLevelAccelerationStructure> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITopLevelAccelerationStructure,
                            IVulkanTopLevelAccelerationStructure,
                            ITopLevelAccelerationStructure,
                            IAccelerationStructure,
                            VulkanAccelerationStructure,
                            IObject)
  static IVulkanTopLevelAccelerationStructure *Create(IVulkanDevice *device,
                                                      const TopLevelAccelerationStructureDesc &desc);
  AccelerationStructureType GetType() const override;
  uint64_t GetCompactedSize() const override;
  const TopLevelAccelerationStructureDesc &GetDesc() const override;
  const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &BottomLevels() const;
  void CommitDependencies(std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references);

 private:
  VulkanTopLevelAccelerationStructureImpl(IVulkanDevice *device, const TopLevelAccelerationStructureDesc &desc);
  TopLevelAccelerationStructureDesc desc_;
  std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> bottom_levels_;

 public:
  ~VulkanTopLevelAccelerationStructureImpl() override {
    DestroyVulkanAccelerationStructure();
  }

 private:
  VulkanAccelerationStructureState state_VulkanAccelerationStructure_;

  VulkanAccelerationStructureState &StateVulkanAccelerationStructure() override {
    return state_VulkanAccelerationStructure_;
  }

  const VulkanAccelerationStructureState &StateVulkanAccelerationStructure() const override {
    return state_VulkanAccelerationStructure_;
  }
};
}  // namespace grassland::rhi::backend::vulkan
