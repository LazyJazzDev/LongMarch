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

namespace grassland::rhi::backend::vulkan {
class VulkanBindingVersionImpl final : public Object<IVulkanBindingVersion> {
 public:
  static VulkanBindingVersionImpl *Create(IVulkanDevice *device, const BindingLayoutDesc &layout);
  ObjectPtr<IVulkanBindingVersion> Clone() const override;
  bool IsInitialized(size_t binding_index, uint32_t element) const override;
  ~VulkanBindingVersionImpl() override;
  VkDescriptorSet Set() const;

 private:
  IVulkanDevice *device_;
  bool WriteDescriptors(size_t binding_index,
                        uint32_t first_element,
                        const std::vector<ITopLevelAccelerationStructure *> &values);
  VulkanBindingVersionImpl(IVulkanDevice *device, const BindingLayoutDesc &layout);
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values);
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values);
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values);
  VkDescriptorSetLayout native_layout_{};
  ObjectPtr<VulkanDescriptorPool::Lease> lease_;
  std::vector<std::vector<bool>> initialized_;

  void InitializeBindingVersion(const BindingLayoutDesc &layout);

 public:
  const BindingLayoutDesc &GetLayout() const override;

  const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const override;

  bool IsComplete() const override;

  bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) override;

  bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) override;

  bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) override;

  bool Write(size_t binding_index,
             uint32_t first_element,
             const std::vector<ITopLevelAccelerationStructure *> &values) override;

 private:
  BindingLayoutDesc layout_;
  // Indexed by sorted layout binding, then array element.
  std::vector<std::vector<ObjectPtr<IObject>>> references_;
};

class VulkanBindingGroupImpl final : public Object<rhi::IBindingGroup, IVulkanBindingGroup> {
 public:
  using Version = IVulkanBindingVersion;
  static IBindingGroup *Create(IVulkanDevice *device, const BindingLayoutDesc &layout);

 private:
  VulkanBindingGroupImpl(IVulkanDevice *device, ObjectPtr<Version> version);

 public:
  ~VulkanBindingGroupImpl() override = default;

 private:
  void InitializeBindingGroup(IVulkanDevice *device, ObjectPtr<IVulkanBindingVersion> version);

 public:
  const BindingLayoutDesc &GetLayout() const override;

  IVulkanDevice *Owner() const override;

  // Borrowed until the next mutation; the context retains it before recording use.
  IVulkanBindingVersion *CurrentVersion() const override;

 private:
  const BindingDesc *FindBinding(uint32_t binding, uint32_t first_element, size_t count, size_t &binding_index) const;

 public:
  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) override;

  bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) override;

  bool BindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0) override;

  bool BindResource(uint32_t binding,
                    const std::vector<ITopLevelAccelerationStructure *> &structures,
                    uint32_t first_element = 0) override;

 private:
  IVulkanDevice *device_;
  BindingLayoutDesc layout_;
  ObjectPtr<IVulkanBindingVersion> version_;
};
}  // namespace grassland::rhi::backend::vulkan
