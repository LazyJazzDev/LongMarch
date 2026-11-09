#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_descriptor_pool.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout);
VkDescriptorSetLayout CreateBindingLayout(VkDevice device, const BindingLayoutDesc &layout);

class VulkanBindingGroup final : public BindingGroup {
 public:
  using IBindingGroup::BindResource;

  void AddRef() noexcept override {
    RetainReference();
  }

  void Release() noexcept override {
    if (ReleaseReference())
      delete this;
  }

  bool BindResource(uint32_t binding,
                    const std::vector<ITopLevelAccelerationStructure *> &structures,
                    uint32_t first_element = 0) override {
    return WriteResources(binding, structures, first_element);
  }

  const BindingLayoutDesc &GetLayout() const override {
    return BindingLayout();
  }

  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) override {
    return WriteResources(binding, buffers, first_element);
  }

  bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) override {
    return WriteResources(binding, images, first_element);
  }

  bool BindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0) override {
    return WriteResources(binding, samplers, first_element);
  }

  class Version final : public BindingGroup::Version {
   public:
    void AddRef() noexcept override {
      RetainReference();
    }

    void Release() noexcept override {
      if (ReleaseReference())
        delete this;
    }

    static Version *Create(VulkanDevice *device, const BindingLayoutDesc &layout);
    ObjectPtr<BindingGroup::Version> Clone() const override;
    bool IsInitialized(size_t binding_index, uint32_t element) const override;
    ~Version() override;
    VkDescriptorSet Set() const;

   private:
    bool WriteDescriptors(size_t binding_index,
                          uint32_t first_element,
                          const std::vector<ITopLevelAccelerationStructure *> &values) override;
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
