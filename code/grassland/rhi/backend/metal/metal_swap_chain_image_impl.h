#pragma once
#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalSwapChainImageImpl final : public Object<rhi::IImage, IMetalSwapChainImage> {
 public:
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
  ~MetalSwapChainImageImpl() override = default;

 private:
  void InitializeImage(ImageDesc desc, IBackendQueue *queue = nullptr);

 public:
  bool SupportsQueue(const IBackendQueue *queue) const override;

  const ImageDesc &GetDesc() const override;

 private:
  ImageDesc desc_{};
  IBackendQueue *queue_{};

  void InitializeMetalImage(ImageDesc desc, const void *device, MTL::Texture *image, IBackendQueue *queue = nullptr);

 public:
  bool SupportsStorage() const override;

  const void *DeviceIdentity() const override;

  MTL::Texture *NativeImage() const override;

 private:
  const void *device_{};
  MTL::Texture *image_{};
};
}  // namespace grassland::rhi::backend::metal
