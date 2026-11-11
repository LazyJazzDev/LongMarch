#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalImage,
                        grassland::rhi::backend::IBackendImage,
                        "3c717669-dee3-407f-92be-d6485832e1c3")

namespace grassland::rhi::backend::metal {

// Native texture is borrowed. Concrete image classes own its backing storage.
class IMetalImage : public IBackendImage {
 public:
  bool SupportsStorage() const override = 0;
  const void *DeviceIdentity() const override = 0;
  virtual MTL::Texture *NativeImage() const = 0;
};

}  // namespace grassland::rhi::backend::metal
