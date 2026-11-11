#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanSwapChainImage, "f9decc34-658f-4a62-bf5b-e7a749c2c44e")

namespace grassland::rhi::backend::vulkan {

class IVulkanSwapChainImage : public IVulkanImage {
 public:
  static IVulkanSwapChainImage *Create(ImageDesc desc, IVulkanWindow *window, VkImage image, VkImageView attachment);

  virtual IVulkanWindow *Owner() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
