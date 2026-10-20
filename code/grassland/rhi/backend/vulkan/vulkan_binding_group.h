#pragma once
#include <mutex>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanDescriptorPool {
  struct Page {
    VkDescriptorPool pool{};
    uint32_t occupied{};
  };

 public:
  class Lease final : public Object<IObject> {
   public:
    Lease(VulkanDescriptorPool *owner, Page *page, VkDescriptorSet set);
    ~Lease() override;
    VkDescriptorSet Set() const;

   private:
    VulkanDescriptorPool *owner_;
    Page *page_;
    VkDescriptorSet set_;
  };

  explicit VulkanDescriptorPool(VkDevice device);
  ~VulkanDescriptorPool();
  ObjectPtr<Lease> Acquire(VkDescriptorSetLayout layout);
  std::pair<size_t, size_t> GetPageCounts() const;

 private:
  void ReleaseSet(Page *page, VkDescriptorSet set);
  VkDevice device_;
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Page>> pages_;
};

bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout);
VkDescriptorSetLayout CreateBindingLayout(VkDevice device, const BindingLayoutDesc &layout);

class VulkanBindingGroup final : public BindingGroup {
 public:
  static VulkanBindingGroup *Create(VulkanDevice *device,
                                    const BindingLayoutDesc &layout,
                                    const std::vector<BindingEntry> &entries);
  ~VulkanBindingGroup() override;
  VkDescriptorSet Set() const;

 private:
  VulkanBindingGroup(VulkanDevice *device, const BindingLayoutDesc &layout, const std::vector<BindingEntry> &entries);
  VkDescriptorSetLayout native_layout_{};
  ObjectPtr<VulkanDescriptorPool::Lease> lease_;
  std::vector<VkImageView> views_;
  std::vector<VkSampler> samplers_;
};
}  // namespace grassland::rhi::backend::vulkan
