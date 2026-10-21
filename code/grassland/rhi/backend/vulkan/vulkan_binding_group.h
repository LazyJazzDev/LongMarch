#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_descriptor_pool.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout);
VkDescriptorSetLayout CreateBindingLayout(VkDevice device, const BindingLayoutDesc &layout);

class VulkanBindingGroup final : public BindingGroup {
 public:
  class Version final : public BindingGroup::Version {
   public:
    static Version *Create(VulkanDevice *device, const BindingLayoutDesc &layout);
    ObjectPtr<BindingGroup::Version> Clone() const override;
    bool IsInitialized(size_t binding_index, uint32_t element) const override;
    ~Version() override;
    VkDescriptorSet Set() const;

   private:
    Version(VulkanDevice *device, const BindingLayoutDesc &layout);
    bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) override;
    bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) override;
    bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) override;
    VkDescriptorSetLayout native_layout_{};
    ObjectPtr<VulkanDescriptorPool::Lease> lease_;
    std::vector<std::vector<bool>> initialized_;
  };

  static VulkanBindingGroup *Create(VulkanDevice *device, const BindingLayoutDesc &layout);

 private:
  VulkanBindingGroup(VulkanDevice *device, ObjectPtr<Version> version);
};
}  // namespace grassland::rhi::backend::vulkan
