#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanSwapChainImage::VulkanSwapChainImage(ImageDesc desc, VulkanWindow *window, uint64_t generation)
    : VulkanImage(desc, window->DeviceIdentity(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
      window_(window),
      generation_(generation) {
}

bool VulkanSwapChainImage::Acquire() {
  const auto result = window_->AcquireNative(sync_index_, acquired_, index_, image_, rendered_);
  if (result == VK_ERROR_OUT_OF_DATE_KHR)
    return false;
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    return Check(result, "vkAcquireNextImageKHR");
  acquired_image_ = true;
  return true;
}

VulkanSwapChainImage::~VulkanSwapChainImage() = default;

bool VulkanSwapChainImage::IsValid() const {
#if !defined(NDEBUG)
  if (present_requested_)
    return false;
#endif
  return acquired_image_ && window_->IsFrameValid(generation_);
}

VkImage VulkanSwapChainImage::NativeImage() const {
  return image_;
}

VkSemaphore VulkanSwapChainImage::AcquireSemaphore() const {
  return acquire_wait_pending_ ? acquired_ : VK_NULL_HANDLE;
}

void VulkanSwapChainImage::ConsumeAcquireSemaphore(VulkanCommandContext *submission) {
  window_->TrackAcquireWait(sync_index_, submission);
  acquire_wait_pending_ = false;
}

VkSemaphore VulkanSwapChainImage::RenderSemaphore() const {
  return rendered_;
}

VkResult VulkanSwapChainImage::Present() {
#if !defined(NDEBUG)
  present_requested_ = true;
#endif
  return window_->PresentFrame(generation_, index_, rendered_);
}

}  // namespace grassland::rhi::backend::vulkan
