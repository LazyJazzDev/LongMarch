#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"

#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {

MetalSwapChainImage::MetalSwapChainImage(ImageDesc desc,
                                         MetalWindow *window,
                                         uint64_t generation,
                                         CA::MetalDrawable *drawable)
    : MetalImage(desc, window->DeviceIdentity(), drawable->texture()),
      window_(window),
      generation_(generation),
      drawable_(NS::RetainPtr(drawable)) {
}

bool MetalSwapChainImage::IsValid() const {
#if !defined(NDEBUG)
  if (present_requested_)
    return false;
#endif
  return window_->IsFrameValid(generation_);
}

CA::MetalDrawable *MetalSwapChainImage::NativeDrawable() const {
  return drawable_.get();
}

void MetalSwapChainImage::DidPresent() {
#if !defined(NDEBUG)
  present_requested_ = true;
#endif
  // presentDrawable and retained-reference command buffers own native GPU use.
  // Invalid caller-held frames must not exhaust CAMetalLayer's drawable pool.
  drawable_.reset();
}

}  // namespace grassland::rhi::backend::metal
