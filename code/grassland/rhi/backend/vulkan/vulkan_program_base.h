#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class VulkanProgramBase;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::VulkanProgramBase)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
// Layout ownership only; the owning Device must outlive the Program.
class VulkanProgramBase : public IObject {
 public:
  VkPipelineLayout NativeLayout() const;

 protected:
  VkDevice LayoutDevice() const {
    return layout_device_;
  }

  explicit VulkanProgramBase(VkDevice device);
  ~VulkanProgramBase();
  VulkanProgramBase(const VulkanProgramBase &) = delete;
  VulkanProgramBase &operator=(const VulkanProgramBase &) = delete;
  bool CreateLayout(const PipelineLayoutDesc &desc);

 private:
  VkDevice layout_device_;
  std::vector<VkDescriptorSetLayout> set_layouts_;
  VkPipelineLayout layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
