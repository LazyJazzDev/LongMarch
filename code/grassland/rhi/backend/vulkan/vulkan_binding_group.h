#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_descriptor_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout);
VkDescriptorSetLayout CreateBindingLayout(VkDevice device, const BindingLayoutDesc &layout);

class IVulkanBindingVersion : public IBackendBindingVersion {
 public:
  static IVulkanBindingVersion *Create(IVulkanDevice *device, const BindingLayoutDesc &layout);

  virtual VkDescriptorSet Set() const = 0;
};

IBindingGroup *CreateVulkanBindingGroup(IVulkanDevice *device, const BindingLayoutDesc &layout);

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanBindingVersion, "fe3339ef-20d4-4c9c-b124-60c9431ce9b8")
