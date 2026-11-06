#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanSwapChainImage::VulkanSwapChainImage(ImageDesc desc, VulkanWindow *window, VkImage image, VkImageView attachment)
    : VulkanImage(desc, window->DeviceIdentity(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
      window_(window),
      image_(image),
      attachment_(attachment) {
}

VulkanSwapChainImage::~VulkanSwapChainImage() = default;

VkImageView VulkanSwapChainImage::AttachmentView() const {
  return attachment_;
}

VkImageView VulkanSwapChainImage::BindingView() const {
  return VK_NULL_HANDLE;
}

bool VulkanSwapChainImage::SupportsQueue(const Queue *queue) const {
  return queue == window_->PresentationQueue();
}

bool VulkanSwapChainImage::IsValid() const {
  return image_ != VK_NULL_HANDLE;
}

VkImage VulkanSwapChainImage::NativeImage() const {
  return image_;
}

}  // namespace grassland::rhi::backend::vulkan
