#include "grassland/rhi/backend/vulkan/vulkan_image.h"

#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanImage::VulkanImage(ImageDesc desc, VulkanWindow *window, uint64_t generation, VkDevice device)
    : Image(desc),
      window_(window),
      generation_(generation),
      device_(device) {
  window_->AcquireFrameLease();
}

bool VulkanImage::Acquire() {
  VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (!Check(vkCreateSemaphore(device_, &semaphore, nullptr, &acquired_), "Create acquire semaphore") ||
      !Check(vkCreateFence(device_, &fence, nullptr, &acquire_fence_), "Create acquire fence"))
    return false;
  const auto result = window_->AcquireNative(acquired_, acquire_fence_, index_, image_, rendered_);
  if (result == VK_ERROR_OUT_OF_DATE_KHR)
    return false;
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    return Check(result, "vkAcquireNextImageKHR");
  acquired_image_ = true;
  return true;
}

VulkanImage::~VulkanImage() {
  if (acquired_image_) {
    while (vkWaitForFences(device_, 1, &acquire_fence_, VK_TRUE, 50000000) == VK_TIMEOUT) {
    }
    if (!presented_)
      window_->CancelFrame(generation_);
  }
  if (acquired_)
    vkDestroySemaphore(device_, acquired_, nullptr);
  if (acquire_fence_)
    vkDestroyFence(device_, acquire_fence_, nullptr);
  window_->ReleaseFrameLease();
}

bool VulkanImage::IsValid() const {
  return acquired_image_ && !presented_ && window_->IsFrameValid(generation_);
}

const void *VulkanImage::DeviceIdentity() const {
  return window_->DeviceIdentity();
}

VkImage VulkanImage::NativeImage() const {
  return image_;
}

VkSemaphore VulkanImage::AcquireSemaphore() const {
  return acquired_;
}

VkSemaphore VulkanImage::RenderSemaphore() const {
  return rendered_;
}

VkResult VulkanImage::Present() {
  presented_ = true;
  return window_->PresentFrame(generation_, index_, rendered_);
}

}  // namespace grassland::rhi::backend::vulkan
