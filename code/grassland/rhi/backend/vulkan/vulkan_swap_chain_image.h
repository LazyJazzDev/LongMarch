#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanSwapChainImage)
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class IVulkanSwapChainImage : public IVulkanImage {
 public:
  static IVulkanSwapChainImage *Create(ImageDesc desc, IVulkanWindow *window, VkImage image, VkImageView attachment);
  virtual bool IsValid() const = 0;
  virtual VkImage NativeImage() const = 0;
  virtual VkImageView AttachmentView() const = 0;
  virtual VkImageView BindingView() const = 0;
  virtual IVulkanWindow *Owner() const = 0;
};

;

}  // namespace grassland::rhi::backend::vulkan
