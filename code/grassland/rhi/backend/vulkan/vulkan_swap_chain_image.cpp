#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanSwapChainImage::VulkanSwapChainImage(ImageDesc desc,
                                           VulkanWindow *window,
                                           VkImage image,
                                           std::shared_ptr<Views> views)
    : VulkanImage(desc, window->DeviceIdentity(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
      window_(window),
      image_(image) {
  views_ = std::move(views);
}

VulkanSwapChainImage::~VulkanSwapChainImage() = default;

bool VulkanSwapChainImage::IsValid() const {
  return image_ != VK_NULL_HANDLE;
}

VkImage VulkanSwapChainImage::NativeImage() const {
  return image_;
}

VulkanWindow *VulkanSwapChainImage::Owner() const {
  return window_;
}

}  // namespace grassland::rhi::backend::vulkan
