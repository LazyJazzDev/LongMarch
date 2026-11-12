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

IMetalSwapChainImage *IMetalSwapChainImage::Create(ImageDesc desc,
                                                   IMetalWindow *window,
                                                   uint64_t generation,
                                                   CA::MetalDrawable *drawable) {
  return new MetalSwapChainImageImpl(desc, window, generation, drawable);
}

void MetalSwapChainImageImpl::InitializeImage(ImageDesc desc, IMetalQueue *queue) {
  desc_ = desc;
  queue_ = queue;
}

bool MetalSwapChainImageImpl::SupportsQueue(const IMetalQueue *queue) const {
  return !queue_ || queue_ == queue;
}

const ImageDesc &MetalSwapChainImageImpl::GetDesc() const {
  return desc_;
}

void MetalSwapChainImageImpl::InitializeMetalImage(ImageDesc desc,
                                                   const void *device,
                                                   MTL::Texture *image,
                                                   IMetalQueue *queue) {
  InitializeImage(desc, queue);
  device_ = device;
  image_ = image;
}

bool MetalSwapChainImageImpl::SupportsStorage() const {
  return (image_->usage() & MTL::TextureUsageShaderWrite) != 0;
}

const void *MetalSwapChainImageImpl::DeviceIdentity() const {
  return device_;
}

MTL::Texture *MetalSwapChainImageImpl::NativeImage() const {
  return image_;
}
}  // namespace grassland::rhi::backend::metal
