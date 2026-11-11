#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_descriptor_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanBindingGroup)

namespace grassland::rhi::backend::vulkan {
bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout);
VkDescriptorSetLayout CreateBindingLayout(VkDevice device, const BindingLayoutDesc &layout);

class IVulkanBindingVersion : public IBackendBindingVersion {
 public:
  static IVulkanBindingVersion *Create(IVulkanDevice *device, const BindingLayoutDesc &layout);
  virtual ObjectPtr<IBackendBindingVersion> Clone() const = 0;
  virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
  virtual VkDescriptorSet Set() const = 0;
};

class IVulkanBindingGroup : public IBackendBindingGroup {
 public:
  using Version = IVulkanBindingVersion;
  static IVulkanBindingGroup *Create(IVulkanDevice *device, const BindingLayoutDesc &layout);
};

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanBindingVersion)
