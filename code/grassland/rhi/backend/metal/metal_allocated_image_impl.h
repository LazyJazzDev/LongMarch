#pragma once
#include "grassland/rhi/backend/metal/metal_allocated_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

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
  ~MetalAllocatedImageImpl() override = default;

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
};
}  // namespace grassland::rhi::backend::metal
