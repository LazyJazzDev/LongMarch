#pragma once
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {
class MetalWindow;

class MetalSwapChainImage final : public MetalImage {
 public:
  MetalSwapChainImage(ImageDesc desc, MetalWindow *window, uint64_t generation, CA::MetalDrawable *drawable);
  bool IsValid() const override;
  CA::MetalDrawable *NativeDrawable() const;
  void DidPresent();

 private:
  MetalWindow *window_;
  uint64_t generation_;
  NS::SharedPtr<CA::MetalDrawable> drawable_;
  bool presented_{};
};

}  // namespace grassland::rhi::backend::metal
