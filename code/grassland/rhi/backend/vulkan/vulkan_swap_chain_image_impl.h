#pragma once
#include "grassland/rhi/backend/backend_image_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/vulkan/vulkan_image_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

namespace grassland::rhi::backend::vulkan {
class VulkanSwapChainImageImpl final : public Object<IVulkanSwapChainImage> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, IVulkanSwapChainImage, IVulkanImage, IBackendImage, rhi::IImage)
  VulkanSwapChainImageImpl(ImageDesc desc, IVulkanWindow *window, VkImage image, VkImageView attachment);
  ~VulkanSwapChainImageImpl() override;
  bool IsValid() const override;
  VkImage NativeImage() const override;
  VkImageView AttachmentView() const override;
  VkImageView BindingView() const override;
  IVulkanWindow *Owner() const;

 private:
  IVulkanWindow *window_;
  VkImage image_{};
  VkImageView attachment_{};

 private:
  VulkanImageState state_VulkanImage_;

  VulkanImageState &StateVulkanImage() override {
    return state_VulkanImage_;
  }

  const VulkanImageState &StateVulkanImage() const override {
    return state_VulkanImage_;
  }

 private:
  ImageState state_Image_;

  ImageState &StateImage() override {
    return state_Image_;
  }

  const ImageState &StateImage() const override {
    return state_Image_;
  }

 public:
  const void *DeviceIdentity() const override {
    return IVulkanImage::CommonDeviceIdentity();
  }

  const ImageDesc &GetDesc() const override {
    return IBackendImage::CommonGetDesc();
  }

  bool SupportsStorage() const override {
    return IBackendImage::CommonSupportsStorage();
  }
};
}  // namespace grassland::rhi::backend::vulkan
