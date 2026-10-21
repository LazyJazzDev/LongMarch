#include "grassland/rhi/backend/vulkan/vulkan_image.h"

#include <utility>

namespace grassland::rhi::backend::vulkan {

VulkanImage::VulkanImage(ImageDesc desc, const void *device, VkImageLayout default_layout)
    : Image(desc),
      device_(device),
      default_layout_(default_layout) {
}

VulkanImage::Views::Views(Views &&other) noexcept
    : device_(other.device_),
      attachment_(std::exchange(other.attachment_, VK_NULL_HANDLE)),
      binding_(std::exchange(other.binding_, VK_NULL_HANDLE)) {
}

VulkanImage::Views &VulkanImage::Views::operator=(Views &&other) noexcept {
  if (this != &other) {
    Reset();
    device_ = other.device_;
    attachment_ = std::exchange(other.attachment_, VK_NULL_HANDLE);
    binding_ = std::exchange(other.binding_, VK_NULL_HANDLE);
  }
  return *this;
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

bool VulkanImage::Views::Initialize(VkDevice device, VkImage image, ImageFormat format, bool sampled) {
  Reset();
  device_ = device;
  VkImageViewCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  create.image = image;
  create.viewType = VK_IMAGE_VIEW_TYPE_2D;
  create.format = NativeImageFormat(format);
  create.subresourceRange = {ImageAspect(format), 0, 1, 0, 1};
  if (!Check(vkCreateImageView(device, &create, nullptr, &attachment_), "Create attachment view"))
    return false;
  if (sampled) {
    if (IsDepthFormat(format) && (create.subresourceRange.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)) {
      create.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      if (!Check(vkCreateImageView(device, &create, nullptr, &binding_), "Create depth binding view")) {
        Reset();
        return false;
      }
    } else {
      binding_ = attachment_;
    }
  }
  return true;
}

VkImageLayout VulkanImage::DefaultLayout() const {
  return default_layout_;
}

const void *VulkanImage::DeviceIdentity() const {
  return device_;
}

}  // namespace grassland::rhi::backend::vulkan
