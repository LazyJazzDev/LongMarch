#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalImage;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalImage)
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend::metal {

// Native texture is borrowed. Concrete image classes own its backing storage.
struct MetalImageState;

class MetalImage : public Image {
 public:
  bool SupportsStorage() const override;
  const void *DeviceIdentity() const override;
  MTL::Texture *NativeImage() const;

 protected:
  void InitializeMetalImage(ImageDesc desc, const void *device, MTL::Texture *image, Queue *queue = nullptr);
  virtual MetalImageState &StateMetalImage() = 0;
  virtual const MetalImageState &StateMetalImage() const = 0;
};

}  // namespace grassland::rhi::backend::metal
