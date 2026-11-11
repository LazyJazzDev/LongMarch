#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base_state.h"

namespace grassland::rhi::backend::vulkan {
void VulkanProgramBase::InitializeVulkanProgramBase(VkDevice device) {
  StateVulkanProgramBase().layout_device_ = device;
}

void VulkanProgramBase::DestroyVulkanProgramBase() {
  if (StateVulkanProgramBase().layout_)
    vkDestroyPipelineLayout(StateVulkanProgramBase().layout_device_, StateVulkanProgramBase().layout_, nullptr);
  for (auto layout : StateVulkanProgramBase().set_layouts_)
    vkDestroyDescriptorSetLayout(StateVulkanProgramBase().layout_device_, layout, nullptr);
}

VkPipelineLayout VulkanProgramBase::NativeLayout() const {
  return StateVulkanProgramBase().layout_;
}

bool VulkanProgramBase::CreateLayout(const PipelineLayoutDesc &desc) {
  StateVulkanProgramBase().set_layouts_.reserve(desc.groups.size());
  for (const auto &group : desc.groups) {
    auto layout = CreateBindingLayout(StateVulkanProgramBase().layout_device_, group);
    if (!layout)
      return false;
    StateVulkanProgramBase().set_layouts_.push_back(layout);
  }
  VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  create.setLayoutCount = uint32_t(StateVulkanProgramBase().set_layouts_.size());
  create.pSetLayouts = StateVulkanProgramBase().set_layouts_.data();
  return Check(vkCreatePipelineLayout(StateVulkanProgramBase().layout_device_, &create, nullptr,
                                      &StateVulkanProgramBase().layout_),
               "Create pipeline layout");
}
}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanProgramBase, "267525b2-441a-4f53-a2e2-826fd05cd6d2")

namespace grassland::rhi::backend::vulkan {
VulkanProgramBaseState::VulkanProgramBaseState() = default;
VulkanProgramBaseState::~VulkanProgramBaseState() = default;
}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
VkDevice VulkanProgramBase::LayoutDevice() const {
  return StateVulkanProgramBase().layout_device_;
}
}  // namespace grassland::rhi::backend::vulkan
