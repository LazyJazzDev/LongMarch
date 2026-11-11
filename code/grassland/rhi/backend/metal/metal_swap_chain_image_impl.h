#pragma once
#include "grassland/rhi/backend/backend_image_state.h"
#include "grassland/rhi/backend/metal/metal_image_state.h"
#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalSwapChainImageImpl final : public Object<IMetalSwapChainImage> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, IMetalSwapChainImage, IMetalImage, IBackendImage, rhi::IImage)
  MetalSwapChainImageImpl(ImageDesc desc, IMetalWindow *window, uint64_t generation, CA::MetalDrawable *drawable);
  bool IsValid() const override;
  CA::MetalDrawable *NativeDrawable() const;
  void DidPresent();

 private:
  IMetalWindow *window_;
  uint64_t generation_;
  NS::SharedPtr<CA::MetalDrawable> drawable_;
#if !defined(NDEBUG)
  bool present_requested_{};
#endif

 public:
  ~MetalSwapChainImageImpl() override {
  }

 private:
  MetalImageState state_MetalImage_;

  MetalImageState &StateMetalImage() override {
    return state_MetalImage_;
  }

  const MetalImageState &StateMetalImage() const override {
    return state_MetalImage_;
  }

 private:
  ImageState state_Image_;

  ImageState &StateImage() override {
    return state_Image_;
  }

  const ImageState &StateImage() const override {
    return state_Image_;
  }

 public:
  bool SupportsStorage() const override {
    return IMetalImage::CommonSupportsStorage();
  }

  const void *DeviceIdentity() const override {
    return IMetalImage::CommonDeviceIdentity();
  }

  const ImageDesc &GetDesc() const override {
    return IBackendImage::CommonGetDesc();
  }
};
}  // namespace grassland::rhi::backend::metal
