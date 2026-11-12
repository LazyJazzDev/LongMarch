#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanSwapChainImageImpl final : public Object<rhi::IImage, IVulkanSwapChainImage> {
 public:
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

  void InitializeImage(ImageDesc desc, IVulkanQueue *queue = nullptr);

 public:
  bool SupportsQueue(const IVulkanQueue *queue) const override;

  bool SupportsStorage() const override;

  const ImageDesc &GetDesc() const override;

 private:
  ImageDesc desc_{};
  IVulkanQueue *queue_{};

  void InitializeVulkanImage(ImageDesc desc,
                             const void *device,
                             VkImageLayout default_layout,
                             IVulkanQueue *queue = nullptr);

 public:
  VkImageAspectFlags Aspect() const override;
  VkImageLayout DefaultLayout() const override;

  const void *DeviceIdentity() const override;

 private:
  const void *device_{};
  VkImageLayout default_layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
