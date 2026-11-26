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
    sampled_ = std::exchange(other.sampled_, false);
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
  if (attachment_)
    vkDestroyImageView(device_, attachment_, nullptr);
  attachment_ = VK_NULL_HANDLE;
  sampled_ = false;
}

VkImageView VulkanImageViews::Attachment() const {
  return attachment_;
}

VkImageView VulkanImageViews::Binding(uint32_t base_mip, uint32_t mip_count) {
  if (!attachment_ || !sampled_)
    return VK_NULL_HANDLE;
#if !defined(NDEBUG)
  if (base_mip >= mip_levels_)
    return VK_NULL_HANDLE;
#endif
  if (!mip_count)
    mip_count = mip_levels_ - base_mip;
#if !defined(NDEBUG)
  if (mip_count > mip_levels_ - base_mip)
    return VK_NULL_HANDLE;
#endif
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
  sampled_ = sampled;
  return true;
}

}  // namespace grassland::rhi::backend::vulkan
