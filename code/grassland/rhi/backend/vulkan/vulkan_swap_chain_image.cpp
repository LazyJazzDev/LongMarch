#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanSwapChainImage::VulkanSwapChainImage(ImageDesc desc, VulkanWindow *window, uint64_t generation, VkImage image)
    : VulkanImage(desc, window->DeviceIdentity(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
      window_(window),
      generation_(generation),
      image_(image) {
}

VulkanSwapChainImage::~VulkanSwapChainImage() = default;

bool VulkanSwapChainImage::IsValid() const {
  return window_->IsFrameValid(generation_);
}

VkImage VulkanSwapChainImage::NativeImage() const {
  return image_;
}

VulkanWindow *VulkanSwapChainImage::Owner() const {
  return window_;
}

}  // namespace grassland::rhi::backend::vulkan
