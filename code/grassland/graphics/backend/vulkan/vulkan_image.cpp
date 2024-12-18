#include "grassland/graphics/backend/vulkan/vulkan_image.h"

namespace grassland::graphics::backend {

VulkanImage::VulkanImage(VulkanCore *core,
                         int width,
                         int height,
                         ImageFormat format)
    : core_(core), format_(format) {
  VkExtent2D extent;
  extent.width = width;
  extent.height = height;
  core_->Device()->CreateImage(ImageFormatToVkFormat(format), extent, &image_);
}

Extent2D VulkanImage::Extent() const {
  auto extent = image_->Extent();
  return {extent.width, extent.height};
}

ImageFormat VulkanImage::Format() const {
  return format_;
}

}  // namespace grassland::graphics::backend
