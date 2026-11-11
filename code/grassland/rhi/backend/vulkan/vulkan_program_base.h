#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanProgramBase, "267525b2-441a-4f53-a2e2-826fd05cd6d2")

namespace grassland::rhi::backend::vulkan {
// Layout ownership only; the owning Device must outlive the Program.
class IVulkanProgramBase : public IObject {
 public:
  virtual VkPipelineLayout NativeLayout() const = 0;
};
}  // namespace grassland::rhi::backend::vulkan
