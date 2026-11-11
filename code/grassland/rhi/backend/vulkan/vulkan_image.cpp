#include "grassland/rhi/backend/vulkan/vulkan_image.h"

#include <utility>

#include "grassland/rhi/backend/vulkan/vulkan_image_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_image_views.h"

namespace grassland::rhi::backend::vulkan {

void IVulkanImage::InitializeVulkanImage(ImageDesc desc,
                                         const void *device,
                                         VkImageLayout default_layout,
                                         IBackendQueue *queue) {
  InitializeImage(desc, queue);
  StateVulkanImage().device_ = device;
  StateVulkanImage().default_layout_ = default_layout;
}

VulkanImageViews::VulkanImageViews(VulkanImageViews &&other) noexcept
    : device_(other.device_),
      attachment_(std::exchange(other.attachment_, VK_NULL_HANDLE)),
      binding_(std::exchange(other.binding_, VK_NULL_HANDLE)) {
}

VulkanImageViews &VulkanImageViews::operator=(VulkanImageViews &&other) noexcept {
  if (this != &other) {
    Reset();
    device_ = other.device_;
    attachment_ = std::exchange(other.attachment_, VK_NULL_HANDLE);
    binding_ = std::exchange(other.binding_, VK_NULL_HANDLE);
  }
  return *this;
}

VulkanImageViews::~VulkanImageViews() {
  Reset();
}

void VulkanImageViews::Reset() {
  if (binding_ && binding_ != attachment_)
    vkDestroyImageView(device_, binding_, nullptr);
  if (attachment_)
    vkDestroyImageView(device_, attachment_, nullptr);
  attachment_ = binding_ = VK_NULL_HANDLE;
}

VkImageView VulkanImageViews::Attachment() const {
  return attachment_;
}

VkImageView VulkanImageViews::Binding() const {
  return binding_;
}

bool VulkanImageViews::Initialize(VkDevice device, VkImage image, ImageFormat format, bool sampled) {
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

VkImageLayout IVulkanImage::DefaultLayout() const {
  return StateVulkanImage().default_layout_;
}

const void *IVulkanImage::CommonDeviceIdentity() const {
  return StateVulkanImage().device_;
}

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanImage, "bab85247-2604-44fe-b152-1ea15663177a")

namespace grassland::rhi::backend::vulkan {
VulkanImageState::VulkanImageState() = default;
VulkanImageState::~VulkanImageState() = default;
}  // namespace grassland::rhi::backend::vulkan
