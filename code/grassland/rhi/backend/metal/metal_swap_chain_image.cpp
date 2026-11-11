#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"

#include "grassland/rhi/backend/metal/metal_swap_chain_image_impl.h"
#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {

MetalSwapChainImageImpl::MetalSwapChainImageImpl(ImageDesc desc,
                                                 IMetalWindow *window,
                                                 uint64_t generation,
                                                 CA::MetalDrawable *drawable)
    : window_(window),
      generation_(generation),
      drawable_(NS::RetainPtr(drawable)) {
  InitializeMetalImage(desc, window->DeviceIdentity(), drawable->texture(), window->OwnerQueue());
}

bool MetalSwapChainImageImpl::IsValid() const {
#if !defined(NDEBUG)
  if (present_requested_)
    return false;
#endif
  return window_->IsFrameValid(generation_);
}

CA::MetalDrawable *MetalSwapChainImageImpl::NativeDrawable() const {
  return drawable_.get();
}

void MetalSwapChainImageImpl::DidPresent() {
#if !defined(NDEBUG)
  present_requested_ = true;
#endif
  // presentDrawable and retained-reference command buffers own native GPU use.
  // Invalid caller-held frames must not exhaust CAMetalLayer's drawable pool.
  drawable_.reset();
}

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalSwapChainImage, "e121e257-92bc-4748-bbe0-2eff7a5af162")

namespace grassland::rhi::backend::metal {
IMetalSwapChainImage *IMetalSwapChainImage::Create(ImageDesc desc,
                                                   IMetalWindow *window,
                                                   uint64_t generation,
                                                   CA::MetalDrawable *drawable) {
  return new MetalSwapChainImageImpl(desc, window, generation, drawable);
}
}  // namespace grassland::rhi::backend::metal
