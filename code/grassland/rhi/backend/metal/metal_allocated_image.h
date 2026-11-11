#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalAllocatedImage)
#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {
class IMetalAllocatedImage : public IMetalImage {
 public:
  static IMetalAllocatedImage *Create(IMetalDevice *device, const ImageDesc &desc);
  virtual bool IsValid() const = 0;
};

;
}  // namespace grassland::rhi::backend::metal
