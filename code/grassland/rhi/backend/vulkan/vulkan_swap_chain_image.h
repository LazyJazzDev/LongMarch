#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanWindow;

class VulkanSwapChainImage final : public VulkanImage {
 public:
  VulkanSwapChainImage(ImageDesc desc,
                       VulkanWindow *window,
                       VkImage image,
                       VkImageView attachment,
                       VkImageView binding);
  ~VulkanSwapChainImage() override;
  bool IsValid() const override;
  VkImage NativeImage() const override;
  VkImageView AttachmentView() const override;
  VkImageView BindingView() const override;
  VulkanWindow *Owner() const;

 private:
  VulkanWindow *window_;
  VkImage image_{};
  VkImageView attachment_{};
  VkImageView binding_{};
};

}  // namespace grassland::rhi::backend::vulkan
