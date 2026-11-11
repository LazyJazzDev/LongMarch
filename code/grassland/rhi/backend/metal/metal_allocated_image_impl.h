#pragma once
#include "grassland/rhi/backend/backend_image_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/metal/metal_allocated_image.h"
#include "grassland/rhi/backend/metal/metal_image_state.h"

namespace grassland::rhi::backend::metal {
class MetalAllocatedImageImpl final : public Object<IMetalAllocatedImage> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, IMetalAllocatedImage, IMetalImage, IBackendImage, rhi::IImage)
  static IMetalAllocatedImage *Create(IMetalDevice *device, const ImageDesc &desc);
  bool IsValid() const override;

 private:
  MetalAllocatedImageImpl(IMetalDevice *device, ImageDesc desc, NS::SharedPtr<MTL::Texture> texture);
  NS::SharedPtr<MTL::Texture> texture_;

 public:
  ~MetalAllocatedImageImpl() override {
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
