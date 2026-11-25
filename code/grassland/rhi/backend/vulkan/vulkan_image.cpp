#include "grassland/rhi/backend/vulkan/vulkan_image.h"

#include <utility>

#include "grassland/rhi/backend/vulkan/vulkan_image_views.h"

namespace grassland::rhi::backend::vulkan {

VulkanImageViews::VulkanImageViews(VulkanImageViews &&other) noexcept {
  *this = std::move(other);
}

VulkanImageViews &VulkanImageViews::operator=(VulkanImageViews &&other) noexcept {
  // Like destruction, relocation requires exclusive ownership of the image.
  if (this != &other) {
    Reset();
    device_ = other.device_;
    image_ = other.image_;
    format_ = other.format_;
    mip_levels_ = other.mip_levels_;
    attachment_ = std::exchange(other.attachment_, VK_NULL_HANDLE);
    binding_ = std::exchange(other.binding_, VK_NULL_HANDLE);
    binding_views_.swap(other.binding_views_);
  }
  return *this;
}

VulkanImageViews::~VulkanImageViews() {
  Reset();
}

void VulkanImageViews::Reset() {
  for (const auto &[range, view] : binding_views_)
    vkDestroyImageView(device_, view, nullptr);
  binding_views_.clear();
  if (binding_ && binding_ != attachment_)
    vkDestroyImageView(device_, binding_, nullptr);
  if (attachment_)
    vkDestroyImageView(device_, attachment_, nullptr);
  attachment_ = binding_ = VK_NULL_HANDLE;
}

VkImageView VulkanImageViews::Attachment() const {
  return attachment_;
}

VkImageView VulkanImageViews::Binding(uint32_t base_mip, uint32_t mip_count) const {
  if (!binding_ || base_mip >= mip_levels_)
    return VK_NULL_HANDLE;
  if (!mip_count)
    mip_count = mip_levels_ - base_mip;
  if (mip_count > mip_levels_ - base_mip)
    return VK_NULL_HANDLE;
  if (!base_mip && mip_count == mip_levels_)
    return binding_;
  if (!base_mip && mip_count == 1 && !(ImageAspect(format_) & VK_IMAGE_ASPECT_STENCIL_BIT))
    return attachment_;
  const auto key = std::make_pair(base_mip, mip_count);
  if (auto found = binding_views_.find(key); found != binding_views_.end())
    return found->second;
  VkImageViewCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  create.image = image_;
  create.viewType = VK_IMAGE_VIEW_TYPE_2D;
  create.format = NativeImageFormat(format_);
  create.subresourceRange = {
      VkImageAspectFlags(IsDepthFormat(format_) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), base_mip,
      mip_count, 0, 1};
  VkImageView view{};
  if (!Check(vkCreateImageView(device_, &create, nullptr, &view), "Create cached mip binding view"))
    return VK_NULL_HANDLE;
  binding_views_.emplace(key, view);
  return view;
}

bool VulkanImageViews::Initialize(VkDevice device,
                                  VkImage image,
                                  ImageFormat format,
                                  bool sampled,
                                  uint32_t mip_levels) {
  Reset();
  device_ = device;
  image_ = image;
  format_ = format;
  mip_levels_ = mip_levels;
  VkImageViewCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  create.image = image;
  create.viewType = VK_IMAGE_VIEW_TYPE_2D;
  create.format = NativeImageFormat(format);
  create.subresourceRange = {ImageAspect(format), 0, 1, 0, 1};
  if (!Check(vkCreateImageView(device, &create, nullptr, &attachment_), "Create attachment view"))
    return false;
  if (sampled) {
    if (mip_levels > 1 ||
        (IsDepthFormat(format) && (create.subresourceRange.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT))) {
      if (IsDepthFormat(format))
        create.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      create.subresourceRange.levelCount = mip_levels;
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

}  // namespace grassland::rhi::backend::vulkan
