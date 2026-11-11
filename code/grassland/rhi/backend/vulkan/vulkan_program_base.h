#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanProgramBase)

namespace grassland::rhi::backend::vulkan {
// Layout ownership only; the owning Device must outlive the Program.
struct VulkanProgramBaseState;

class IVulkanProgramBase : public IObject {
 public:
  VkPipelineLayout NativeLayout() const;

 protected:
  VkDevice LayoutDevice() const;
  void InitializeVulkanProgramBase(VkDevice device);
  void DestroyVulkanProgramBase();
  bool CreateLayout(const PipelineLayoutDesc &desc);
  virtual VulkanProgramBaseState &StateVulkanProgramBase() = 0;
  virtual const VulkanProgramBaseState &StateVulkanProgramBase() const = 0;
};
}  // namespace grassland::rhi::backend::vulkan
