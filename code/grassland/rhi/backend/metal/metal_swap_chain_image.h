#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalSwapChainImage;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalSwapChainImage)
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {
class MetalWindow;

class MetalSwapChainImage final : public MetalImage {
 public:
  LONGMARCH_RHI_QUERY(MetalSwapChainImage, MetalImage)

  MetalSwapChainImage(ImageDesc desc, MetalWindow *window, uint64_t generation, CA::MetalDrawable *drawable);
  bool IsValid() const override;
  CA::MetalDrawable *NativeDrawable() const;
  void DidPresent();

 private:
  MetalWindow *window_;
  uint64_t generation_;
  NS::SharedPtr<CA::MetalDrawable> drawable_;
#if !defined(NDEBUG)
  bool present_requested_{};
#endif
};

}  // namespace grassland::rhi::backend::metal
