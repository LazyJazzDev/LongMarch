#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalImage)

namespace grassland::rhi::backend::metal {

// Native texture is borrowed. Concrete image classes own its backing storage.
struct MetalImageState;

class IMetalImage : public IBackendImage {
 public:
  bool SupportsStorage() const override = 0;
  bool CommonSupportsStorage() const;
  const void *DeviceIdentity() const override = 0;
  const void *CommonDeviceIdentity() const;
  MTL::Texture *NativeImage() const;

 protected:
  void InitializeMetalImage(ImageDesc desc, const void *device, MTL::Texture *image, IBackendQueue *queue = nullptr);
  virtual MetalImageState &StateMetalImage() = 0;
  virtual const MetalImageState &StateMetalImage() const = 0;
};

}  // namespace grassland::rhi::backend::metal
