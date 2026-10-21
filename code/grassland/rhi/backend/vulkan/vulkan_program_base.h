#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
// Layout ownership only; the owning Device must outlive the Program.
class VulkanProgramBase {
 public:
  VkPipelineLayout NativeLayout() const;

 protected:
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
