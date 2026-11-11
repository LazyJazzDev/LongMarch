#pragma once
#include "grassland/rhi/backend/backend_image_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_image_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_image_views.h"

namespace grassland::rhi::backend::vulkan {
class VulkanAllocatedImageImpl final : public Object<IVulkanAllocatedImage> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, IVulkanAllocatedImage, VulkanImage, Image, rhi::IImage)
  static IVulkanAllocatedImage *Create(IVulkanDevice *device, const ImageDesc &desc);
  ~VulkanAllocatedImageImpl() override;
  bool IsValid() const override;
  bool SupportsStorage() const override;
  VkImage NativeImage() const override;
  VkImageView AttachmentView() const override;
  VkImageView BindingView() const override;

 private:
  VulkanAllocatedImageImpl(IVulkanDevice *device, ImageDesc desc);
  IVulkanDevice *device_;
  VkImage image_{};
  VulkanImageViews views_;
  bool storage_{};
  VmaAllocation allocation_{};

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
};
}  // namespace grassland::rhi::backend::vulkan
