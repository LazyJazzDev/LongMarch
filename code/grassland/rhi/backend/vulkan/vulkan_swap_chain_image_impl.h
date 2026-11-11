#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

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

  void InitializeImage(ImageDesc desc, IBackendQueue *queue = nullptr);

 public:
  bool SupportsQueue(const IBackendQueue *queue) const override;

  bool SupportsStorage() const override;

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
  const void *device_{};
  VkImageLayout default_layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
