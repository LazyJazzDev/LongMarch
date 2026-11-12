#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

#include <utility>

#include "grassland/rhi/backend/vulkan/vulkan_image_views.h"
#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image_impl.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanSwapChainImageImpl::VulkanSwapChainImageImpl(ImageDesc desc,
                                                   IVulkanWindow *window,
                                                   VkImage image,
                                                   VkImageView attachment)
    : window_(window),
      image_(image),
      attachment_(attachment) {
  InitializeVulkanImage(desc, window->DeviceIdentity(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, window->OwnerQueue());
}

VulkanSwapChainImageImpl::~VulkanSwapChainImageImpl() {
}

VkImageView VulkanSwapChainImageImpl::AttachmentView() const {
  return attachment_;
}

VkImageView VulkanSwapChainImageImpl::BindingView() const {
  return VK_NULL_HANDLE;
}

bool VulkanSwapChainImageImpl::IsValid() const {
  return image_ != VK_NULL_HANDLE;
}

VkImage VulkanSwapChainImageImpl::NativeImage() const {
  return image_;
}

IVulkanWindow *VulkanSwapChainImageImpl::Owner() const {
  return window_;
}

IVulkanSwapChainImage *IVulkanSwapChainImage::Create(ImageDesc desc,
                                                     IVulkanWindow *window,
                                                     VkImage image,
                                                     VkImageView attachment) {
  return new VulkanSwapChainImageImpl(desc, window, image, attachment);
}

void VulkanSwapChainImageImpl::InitializeImage(ImageDesc desc, IVulkanQueue *queue) {
  desc_ = desc;
  queue_ = queue;
}

bool VulkanSwapChainImageImpl::SupportsQueue(const IVulkanQueue *queue) const {
  return !queue_ || queue_ == queue;
}

bool VulkanSwapChainImageImpl::SupportsStorage() const {
  return false;
}

const ImageDesc &VulkanSwapChainImageImpl::GetDesc() const {
  return desc_;
}

void VulkanSwapChainImageImpl::InitializeVulkanImage(ImageDesc desc,
                                                     const void *device,
                                                     VkImageLayout default_layout,
                                                     IVulkanQueue *queue) {
  InitializeImage(desc, queue);
  device_ = device;
  default_layout_ = default_layout;
}

VkImageAspectFlags VulkanSwapChainImageImpl::Aspect() const {
  return ImageAspect(desc_.format);
}

VkImageLayout VulkanSwapChainImageImpl::DefaultLayout() const {
  return default_layout_;
}

const void *VulkanSwapChainImageImpl::DeviceIdentity() const {
  return device_;
}
}  // namespace grassland::rhi::backend::vulkan
