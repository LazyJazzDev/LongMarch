#include "grassland/rhi/backend/vulkan/vulkan_image.h"

namespace grassland::rhi::backend::vulkan {

VulkanImage::VulkanImage(ImageDesc desc, const void *device) : Image(desc), device_(device) {
}

const void *VulkanImage::DeviceIdentity() const {
  return device_;
}

}  // namespace grassland::rhi::backend::vulkan
