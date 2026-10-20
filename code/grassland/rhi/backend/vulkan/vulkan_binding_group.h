#pragma once
#include <array>
#include <mutex>
#include <unordered_map>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanDescriptorPool {
  // Uniform buffers, storage buffers, sampled images, storage images, samplers.
  using Pattern = std::array<uint32_t, 5>;

  struct PatternHash {
    size_t operator()(const Pattern &pattern) const;
  };

  struct Page {
    VkDescriptorPool pool{};
    Pattern pattern{};
    uint32_t capacity{}, available_sets{};
    // Remaining descriptors of type i = pattern[i] * available_sets.
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
  ObjectPtr<Lease> Acquire(VkDescriptorSetLayout layout, const BindingLayoutDesc &desc);
  std::pair<size_t, size_t> GetPageCounts() const;

 private:
  void ReleaseSet(Page *page, VkDescriptorSet set);
  VkDevice device_;
  mutable std::mutex mutex_;
  std::unordered_map<Pattern, std::vector<std::unique_ptr<Page>>, PatternHash> pages_;
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
