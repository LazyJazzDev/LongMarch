#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_image_views.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanAllocatedImageImpl final : public Object<IVulkanAllocatedImage> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, IVulkanAllocatedImage, IVulkanImage, IBackendImage, rhi::IImage)
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

  void InitializeImage(ImageDesc desc, IBackendQueue *queue = nullptr);

 public:
  bool SupportsQueue(const IBackendQueue *queue) const override;

  const ImageDesc &GetDesc() const override;

 private:
  ImageDesc desc_{};
  IBackendQueue *queue_{};

  void InitializeVulkanImage(ImageDesc desc,
                             const void *device,
                             VkImageLayout default_layout,
                             IBackendQueue *queue = nullptr);

 public:
  VkImageLayout DefaultLayout() const override;

  const void *DeviceIdentity() const override;

 private:
  VkImageLayout default_layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
