#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanSwapChainImage::VulkanSwapChainImage(ImageDesc desc, VulkanWindow *window, uint64_t generation, VkDevice device)
    : VulkanImage(desc, window->DeviceIdentity(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
      window_(window),
      generation_(generation),
      device_(device) {
}

bool VulkanSwapChainImage::Acquire() {
  VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (!Check(vkCreateSemaphore(device_, &semaphore, nullptr, &acquired_), "Create acquire semaphore"))
    return false;
  if (!Check(vkCreateFence(device_, &fence, nullptr, &acquire_fence_), "Create acquire fence"))
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
    while (vkWaitForFences(device_, 1, &acquire_fence_, VK_TRUE, 50000000) == VK_TIMEOUT) {
    }
  }
  if (acquired_)
    vkDestroySemaphore(device_, acquired_, nullptr);
  if (acquire_fence_)
    vkDestroyFence(device_, acquire_fence_, nullptr);
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
