#pragma once
#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDescriptorPool final {
  class LeaseImpl;
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
  class ILease : public IObject {
   public:
    virtual VkDescriptorSet Set() const = 0;
  };

  using Lease = ILease;

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

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::VulkanDescriptorPool::ILease,
                        grassland::rhi::IObject,
                        "4eacb408-4ddb-4df5-9846-c7fd2101efb2")
