#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class IMetalAllocatedImage;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalAllocatedImage)
#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {
class IMetalDevice;

class IMetalAllocatedImage : public MetalImage {
 public:
  static IMetalAllocatedImage *Create(IMetalDevice *device, const ImageDesc &desc);
  virtual bool IsValid() const = 0;
};

;
}  // namespace grassland::rhi::backend::metal
