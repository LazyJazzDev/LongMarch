#pragma once
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
IImage *CreateMetalAllocatedImage(IMetalDevice *device, const ImageDesc &desc);

}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
class MetalAllocatedImageImpl final : public Object<rhi::IImage, IMetalImage> {
 public:
  static IImage *Create(IMetalDevice *device, const ImageDesc &desc);
  bool IsValid() const override;

 private:
  MetalAllocatedImageImpl(IMetalDevice *device, ImageDesc desc, NS::SharedPtr<MTL::Texture> texture);
  NS::SharedPtr<MTL::Texture> texture_;

 public:
  ~MetalAllocatedImageImpl() override = default;

 private:
  void InitializeImage(ImageDesc desc, IMetalQueue *queue = nullptr);

 public:
  bool SupportsQueue(const IMetalQueue *queue) const override;

  const ImageDesc &GetDesc() const override;

 private:
  ImageDesc desc_{};
  IMetalQueue *queue_{};

  void InitializeMetalImage(ImageDesc desc, const void *device, MTL::Texture *image, IMetalQueue *queue = nullptr);

 public:
  bool SupportsStorage() const override;

  const void *DeviceIdentity() const override;

  MTL::Texture *NativeImage() const override;

 private:
  const void *device_{};
};
}  // namespace grassland::rhi::backend::metal
