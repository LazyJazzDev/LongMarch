#pragma once
#include "grassland/rhi/backend/backend_binding_group_bindinggroup_state.h"
#include "grassland/rhi/backend/backend_binding_group_bindingversion_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanBindingVersionImpl final : public Object<IVulkanBindingVersion> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IObject, IVulkanBindingVersion, IBackendBindingVersion, IObject)
  static VulkanBindingVersionImpl *Create(IVulkanDevice *device, const BindingLayoutDesc &layout);
  ObjectPtr<IBackendBindingVersion> Clone() const override;
  bool IsInitialized(size_t binding_index, uint32_t element) const override;
  ~VulkanBindingVersionImpl() override;
  VkDescriptorSet Set() const;

 private:
  IVulkanDevice *device_;
  bool WriteDescriptors(size_t binding_index,
                        uint32_t first_element,
                        const std::vector<ITopLevelAccelerationStructure *> &values) override;
  VulkanBindingVersionImpl(IVulkanDevice *device, const BindingLayoutDesc &layout);
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) override;
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) override;
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) override;
  VkDescriptorSetLayout native_layout_{};
  ObjectPtr<VulkanDescriptorPool::Lease> lease_;
  std::vector<std::vector<bool>> initialized_;

 private:
  BindingVersionState state_BindingVersion_;

  BindingVersionState &StateBindingVersion() override {
    return state_BindingVersion_;
  }

  const BindingVersionState &StateBindingVersion() const override {
    return state_BindingVersion_;
  }
};

class VulkanBindingGroupImpl final : public Object<IVulkanBindingGroup> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IBindingGroup, IVulkanBindingGroup, IBackendBindingGroup, IBindingGroup)
  static IVulkanBindingGroup *Create(IVulkanDevice *device, const BindingLayoutDesc &layout);

 private:
  VulkanBindingGroupImpl(IVulkanDevice *device, ObjectPtr<Version> version);

 public:
  ~VulkanBindingGroupImpl() override {
  }

 private:
  BindingGroupState state_BindingGroup_;

  BindingGroupState &StateBindingGroup() override {
    return state_BindingGroup_;
  }

  const BindingGroupState &StateBindingGroup() const override {
    return state_BindingGroup_;
  }

 public:
  bool BindResource(uint32_t binding,
                    const std::vector<ITopLevelAccelerationStructure *> &structures,
                    uint32_t first_element = 0) override {
    return IBackendBindingGroup::CommonBindResource(binding, structures, first_element);
  }

  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) override {
    return IBackendBindingGroup::CommonBindResource(binding, buffers, first_element);
  }

  bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) override {
    return IBackendBindingGroup::CommonBindResource(binding, images, first_element);
  }

  bool BindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0) override {
    return IBackendBindingGroup::CommonBindResource(binding, samplers, first_element);
  }

  const BindingLayoutDesc &GetLayout() const override {
    return IBackendBindingGroup::CommonGetLayout();
  }
};
}  // namespace grassland::rhi::backend::vulkan
