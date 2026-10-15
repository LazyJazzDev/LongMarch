#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanSwapChainImage::VulkanSwapChainImage(ImageDesc desc,
                                           VulkanWindow *window,
                                           uint64_t generation,
                                           VulkanDevice *device)
    : VulkanImage(desc, window->DeviceIdentity(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
      window_(window),
      generation_(generation),
      device_(device) {
}

bool VulkanSwapChainImage::Acquire() {
  if (!device_->AcquireImageSync(acquired_, acquire_fence_))
    return false;
  const auto result = window_->AcquireNative(acquired_, acquire_fence_, index_, image_, rendered_);
  if (result == VK_ERROR_OUT_OF_DATE_KHR)
    return false;
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    return Check(result, "vkAcquireNextImageKHR");
  acquired_image_ = true;
  return true;
}

VulkanSwapChainImage::~VulkanSwapChainImage() {
  if (acquired_image_) {
    while (vkWaitForFences(device_->NativeDevice(), 1, &acquire_fence_, VK_TRUE, 50000000) == VK_TIMEOUT) {
    }
  }
  // A submitted context keeps this image alive until its acquire wait has completed.
  // An unused acquired image still has a signaled binary semaphore: destroy it instead.
  if (!acquire_wait_pending_) {
    device_->RecycleImageSync(acquired_, acquire_fence_);
    return;
  }
  if (acquired_)
    vkDestroySemaphore(device_->NativeDevice(), acquired_, nullptr);
  if (acquire_fence_)
    vkDestroyFence(device_->NativeDevice(), acquire_fence_, nullptr);
}

bool VulkanSwapChainImage::IsValid() const {
  return acquired_image_ && !presented_ && window_->IsFrameValid(generation_);
}

VkImage VulkanSwapChainImage::NativeImage() const {
  return image_;
}

VkSemaphore VulkanSwapChainImage::AcquireSemaphore() const {
  return acquire_wait_pending_ ? acquired_ : VK_NULL_HANDLE;
}

void VulkanSwapChainImage::ConsumeAcquireSemaphore() {
  acquire_wait_pending_ = false;
}

VkSemaphore VulkanSwapChainImage::RenderSemaphore() const {
  return rendered_;
}

VkResult VulkanSwapChainImage::Present() {
  presented_ = true;
  return window_->PresentFrame(generation_, index_, rendered_);
}

}  // namespace grassland::rhi::backend::vulkan
