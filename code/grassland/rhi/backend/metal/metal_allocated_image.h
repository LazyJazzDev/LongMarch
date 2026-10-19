#pragma once
#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalAllocatedImage final : public MetalImage {
 public:
  static MetalAllocatedImage *Create(MetalDevice *device, const ImageDesc &desc);
  bool IsValid() const override;

 private:
  MetalAllocatedImage(MetalDevice *device, ImageDesc desc, NS::SharedPtr<MTL::Texture> texture);
  NS::SharedPtr<MTL::Texture> texture_;
};
}  // namespace grassland::rhi::backend::metal
