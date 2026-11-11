#pragma once
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalSwapChainImage)

namespace grassland::rhi::backend::metal {
class IMetalSwapChainImage : public IMetalImage {
 public:
  static IMetalSwapChainImage *Create(ImageDesc desc,
                                      IMetalWindow *window,
                                      uint64_t generation,
                                      CA::MetalDrawable *drawable);
  virtual bool IsValid() const = 0;
  virtual CA::MetalDrawable *NativeDrawable() const = 0;
  virtual void DidPresent() = 0;
};

}  // namespace grassland::rhi::backend::metal
