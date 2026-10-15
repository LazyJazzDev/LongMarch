#include "grassland/rhi/backend/vulkan/vulkan_image.h"

namespace grassland::rhi::backend::vulkan {

VulkanImage::VulkanImage(ImageDesc desc, const void *device, VkImageLayout default_layout)
    : Image(desc),
      device_(device),
      default_layout_(default_layout) {
}

VkImageLayout VulkanImage::DefaultLayout() const {
  return default_layout_;
}

const void *VulkanImage::DeviceIdentity() const {
  return device_;
}

}  // namespace grassland::rhi::backend::vulkan
