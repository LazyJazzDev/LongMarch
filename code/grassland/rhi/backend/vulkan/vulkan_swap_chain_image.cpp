#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanSwapChainImage::VulkanSwapChainImage(ImageDesc desc,
                                           VulkanWindow *window,
                                           VkImage image,
                                           bool needs_initialization)
    : VulkanImage(desc, window->DeviceIdentity(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
      window_(window),
      image_(image),
      needs_initialization_(needs_initialization) {
}

VulkanSwapChainImage::~VulkanSwapChainImage() = default;

bool VulkanSwapChainImage::IsValid() const {
  return image_ != VK_NULL_HANDLE;
}

bool VulkanSwapChainImage::NeedsInitialization() const {
  return needs_initialization_;
}

void VulkanSwapChainImage::MarkInitialized() {
  if (needs_initialization_) {
    window_->MarkImageInitialized(image_);
    needs_initialization_ = false;
  }
}

VkImage VulkanSwapChainImage::NativeImage() const {
  return image_;
}

VulkanWindow *VulkanSwapChainImage::Owner() const {
  return window_;
}

}  // namespace grassland::rhi::backend::vulkan
