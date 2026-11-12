#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_descriptor_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout);
VkDescriptorSetLayout CreateBindingLayout(VkDevice device, const BindingLayoutDesc &layout);

class IVulkanBindingVersion : public IObject {
 public:
  virtual bool IsUniquelyOwned() const noexcept = 0;
  virtual const BindingLayoutDesc &GetLayout() const = 0;
  virtual const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const = 0;
  virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
  virtual bool IsComplete() const = 0;
  virtual ObjectPtr<IVulkanBindingVersion> Clone() const = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) = 0;
  virtual bool Write(size_t binding_index,
                     uint32_t first_element,
                     const std::vector<ITopLevelAccelerationStructure *> &values) = 0;
  static IVulkanBindingVersion *Create(IVulkanDevice *device, const BindingLayoutDesc &layout);

  virtual VkDescriptorSet Set() const = 0;
};

class IVulkanBindingGroup : public IObject {
 public:
  virtual IVulkanDevice *Owner() const = 0;
  // Borrowed until the next mutation; retain before recording use.
  virtual IVulkanBindingVersion *CurrentVersion() const = 0;
};

IBindingGroup *CreateVulkanBindingGroup(IVulkanDevice *device, const BindingLayoutDesc &layout);

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanBindingVersion,
                        grassland::rhi::IObject,
                        "fe3339ef-20d4-4c9c-b124-60c9431ce9b8")

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanBindingGroup,
                        grassland::rhi::IObject,
                        "9449da34-a091-4ffa-87f7-7610931636d8")
