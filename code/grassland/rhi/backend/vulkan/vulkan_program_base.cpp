#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"

namespace grassland::rhi::backend::vulkan {
VulkanProgramBase::VulkanProgramBase(VkDevice device) : layout_device_(device) {
}

VulkanProgramBase::~VulkanProgramBase() {
  if (layout_)
    vkDestroyPipelineLayout(layout_device_, layout_, nullptr);
  for (auto layout : set_layouts_)
    vkDestroyDescriptorSetLayout(layout_device_, layout, nullptr);
}

VkPipelineLayout VulkanProgramBase::NativeLayout() const {
  return layout_;
}

bool VulkanProgramBase::CreateLayout(const PipelineLayoutDesc &desc) {
  if (layout_)
    return true;
  // A failed Finalize can be retried; release any partially created set layouts.
  for (auto layout : set_layouts_)
    vkDestroyDescriptorSetLayout(layout_device_, layout, nullptr);
  set_layouts_.clear();
  set_layouts_.reserve(desc.groups.size());
  for (const auto &group : desc.groups) {
    auto layout = CreateBindingLayout(layout_device_, group);
    if (!layout)
      return false;
    set_layouts_.push_back(layout);
  }
  VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  create.setLayoutCount = uint32_t(set_layouts_.size());
  create.pSetLayouts = set_layouts_.data();
  return Check(vkCreatePipelineLayout(layout_device_, &create, nullptr, &layout_), "Create pipeline layout");
}
}  // namespace grassland::rhi::backend::vulkan
