#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {

MetalImage::MetalImage(ImageDesc desc, const void *device, MTL::Texture *image, Queue *queue)
    : Image(desc, queue),
      device_(device),
      image_(image) {
}

bool MetalImage::SupportsStorage() const {
  return (image_->usage() & MTL::TextureUsageShaderWrite) != 0;
}

const void *MetalImage::DeviceIdentity() const {
  return device_;
}

MTL::Texture *MetalImage::NativeImage() const {
  return image_;
}

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalImage, "3c717669-dee3-407f-92be-d6485832e1c3")
