#include "grassland/rhi/backend/vulkan/vulkan_image.h"

namespace grassland::rhi::backend::vulkan {

VulkanImage::VulkanImage(ImageDesc desc, const void *device, VkImageLayout default_layout)
    : Image(desc),
      device_(device),
      default_layout_(default_layout) {
}

VulkanImage::Views::Views(VkDevice device) : device_(device) {
}

VulkanImage::Views::~Views() {
  Reset();
}

void VulkanImage::Views::Reset() {
  if (binding_ && binding_ != attachment_)
    vkDestroyImageView(device_, binding_, nullptr);
  if (attachment_)
    vkDestroyImageView(device_, attachment_, nullptr);
  attachment_ = binding_ = VK_NULL_HANDLE;
}

VkImageView VulkanImage::Views::Attachment() const {
  return attachment_;
}

VkImageView VulkanImage::Views::Binding() const {
  return binding_;
}

std::shared_ptr<VulkanImage::Views> VulkanImage::Views::Create(VkDevice device,
                                                               VkImage image,
                                                               ImageFormat format,
                                                               bool sampled) {
  auto views = std::shared_ptr<Views>(new Views(device));
  VkImageViewCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  create.image = image;
  create.viewType = VK_IMAGE_VIEW_TYPE_2D;
  create.format = NativeImageFormat(format);
  create.subresourceRange = {ImageAspect(format), 0, 1, 0, 1};
  if (!Check(vkCreateImageView(device, &create, nullptr, &views->attachment_), "Create attachment view"))
    return {};
  if (sampled) {
    if (IsDepthFormat(format) && (create.subresourceRange.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)) {
      create.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      if (!Check(vkCreateImageView(device, &create, nullptr, &views->binding_), "Create depth binding view"))
        return {};
    } else {
      views->binding_ = views->attachment_;
    }
  }
  return views;
}

VkImageView VulkanImage::AttachmentView() const {
  return views_->Attachment();
}

VkImageView VulkanImage::BindingView() const {
  return views_->Binding();
}

VkImageLayout VulkanImage::DefaultLayout() const {
  return default_layout_;
}

const void *VulkanImage::DeviceIdentity() const {
  return device_;
}

}  // namespace grassland::rhi::backend::vulkan
