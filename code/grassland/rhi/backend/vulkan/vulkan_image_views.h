#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanImageViews final {
 public:
  VulkanImageViews() = default;
  VulkanImageViews(VulkanImageViews &&other) noexcept;
  VulkanImageViews &operator=(VulkanImageViews &&other) noexcept;
  bool Initialize(VkDevice device, VkImage image, ImageFormat format, bool sampled);
  ~VulkanImageViews();
  VulkanImageViews(const VulkanImageViews &) = delete;
  VulkanImageViews &operator=(const VulkanImageViews &) = delete;
  void Reset();
  VkImageView Attachment() const;
  VkImageView Binding() const;

 private:
  VkDevice device_{};
  VkImageView attachment_{}, binding_{};
};
}  // namespace grassland::rhi::backend::vulkan
