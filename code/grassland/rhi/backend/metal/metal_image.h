#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend::metal {

// Native texture is borrowed. Concrete image classes own its backing storage.
class MetalImage : public Image {
 public:
  const void *DeviceIdentity() const override;
  MTL::Texture *NativeImage() const;

 protected:
  MetalImage(ImageDesc desc, const void *device, MTL::Texture *image);

 private:
  const void *device_;
  MTL::Texture *image_;
};

}  // namespace grassland::rhi::backend::metal
