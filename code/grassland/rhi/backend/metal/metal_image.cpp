#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {

MetalImage::MetalImage(ImageDesc desc, const void *device, MTL::Texture *image)
    : Image(desc),
      device_(device),
      image_(image) {
}

const void *MetalImage::DeviceIdentity() const {
  return device_;
}

MTL::Texture *MetalImage::NativeImage() const {
  return image_;
}

}  // namespace grassland::rhi::backend::metal
