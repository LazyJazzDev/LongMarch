#pragma once
#include <map>

#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanImageViews final {
 public:
  VulkanImageViews() = default;
  VulkanImageViews(VulkanImageViews &&other) noexcept;
  VulkanImageViews &operator=(VulkanImageViews &&other) noexcept;
  bool Initialize(VkDevice device, VkImage image, ImageFormat format, bool sampled, uint32_t mip_levels = 1);
  ~VulkanImageViews();
  VulkanImageViews(const VulkanImageViews &) = delete;
  VulkanImageViews &operator=(const VulkanImageViews &) = delete;
  void Reset();
  VkImageView Attachment() const;
  // Callers serialize binding view access for the same image, including across groups.
  VkImageView Binding(uint32_t base_mip = 0, uint32_t mip_count = 0);

 private:
  VkDevice device_{};
  VkImage image_{};
  ImageFormat format_{};
  uint32_t mip_levels_{};
  // Shared by binding groups; entries live until the owning image is destroyed.
  std::map<std::pair<uint32_t, uint32_t>, VkImageView> binding_views_;
  VkImageView attachment_{};
  bool sampled_{};
};
}  // namespace grassland::rhi::backend::vulkan
