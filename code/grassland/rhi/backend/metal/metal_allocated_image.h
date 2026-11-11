#pragma once
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalAllocatedImage)

namespace grassland::rhi::backend::metal {
class IMetalAllocatedImage : public IMetalImage {
 public:
  static IMetalAllocatedImage *Create(IMetalDevice *device, const ImageDesc &desc);
  virtual bool IsValid() const = 0;
};

}  // namespace grassland::rhi::backend::metal
