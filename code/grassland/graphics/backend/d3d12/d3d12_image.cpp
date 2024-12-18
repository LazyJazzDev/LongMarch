#include "grassland/graphics/backend/d3d12/d3d12_image.h"

namespace grassland::graphics::backend {

D3D12Image::D3D12Image(D3D12Core *core,
                       int width,
                       int height,
                       ImageFormat format)
    : core_(core), format_(format) {
  core_->Device()->CreateImage(width, height, ImageFormatToDXGIFormat(format),
                               &image_);
}

Extent2D D3D12Image::Extent() const {
  Extent2D extent;
  extent.width = image_->Width();
  extent.height = image_->Height();
  return extent;
}

ImageFormat D3D12Image::Format() const {
  return format_;
}

d3d12::Image *D3D12Image::Image() const {
  return image_.get();
}

}  // namespace grassland::graphics::backend
