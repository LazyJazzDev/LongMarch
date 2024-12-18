#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanImage : public Image {
 public:
  VulkanImage(VulkanCore *core, int width, int height, ImageFormat format);
  Extent2D Extent() const override;
  ImageFormat Format() const override;

  vulkan::Image *Image() {
    return image_.get();
  }

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::Image> image_;
  ImageFormat format_;
};

}  // namespace grassland::graphics::backend
