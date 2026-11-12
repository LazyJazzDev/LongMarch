#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_image.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalImage,
                        grassland::rhi::IObject,
                        "3c717669-dee3-407f-92be-d6485832e1c3")

namespace grassland::rhi::backend::metal {

// Native texture is borrowed. Concrete image classes own its backing storage.
class IMetalImage : public IObject {
 public:
  virtual bool SupportsQueue(const IMetalQueue *queue) const = 0;
  virtual bool SupportsStorage() const = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual MTL::Texture *NativeImage() const = 0;
};

}  // namespace grassland::rhi::backend::metal
