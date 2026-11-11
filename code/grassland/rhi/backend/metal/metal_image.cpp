#include "grassland/rhi/backend/metal/metal_image.h"

#include "grassland/rhi/backend/metal/metal_image_state.h"

namespace grassland::rhi::backend::metal {

void MetalImage::InitializeMetalImage(ImageDesc desc, const void *device, MTL::Texture *image, Queue *queue) {
  InitializeImage(desc, queue);
  StateMetalImage().device_ = device;
  StateMetalImage().image_ = image;
}

bool MetalImage::SupportsStorage() const {
  return (StateMetalImage().image_->usage() & MTL::TextureUsageShaderWrite) != 0;
}

const void *MetalImage::DeviceIdentity() const {
  return StateMetalImage().device_;
}

MTL::Texture *MetalImage::NativeImage() const {
  return StateMetalImage().image_;
}

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalImage, "3c717669-dee3-407f-92be-d6485832e1c3")

namespace grassland::rhi::backend::metal {
MetalImageState::MetalImageState() = default;
MetalImageState::~MetalImageState() = default;
}  // namespace grassland::rhi::backend::metal
