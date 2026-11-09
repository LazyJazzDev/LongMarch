#pragma once
#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDescriptorPool {
  // Uniform buffers, storage buffers, sampled images, storage images, samplers, acceleration structures.
  using Pattern = std::array<uint32_t, 6>;

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

}  // namespace grassland::rhi::backend::vulkan
