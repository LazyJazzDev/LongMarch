#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

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

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanSwapChainImage, "f9decc34-658f-4a62-bf5b-e7a749c2c44e")

namespace grassland::rhi::backend::vulkan {
IVulkanSwapChainImage *IVulkanSwapChainImage::Create(ImageDesc desc,
                                                     IVulkanWindow *window,
                                                     VkImage image,
                                                     VkImageView attachment) {
  return new VulkanSwapChainImageImpl(desc, window, image, attachment);
}
}  // namespace grassland::rhi::backend::vulkan
