#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalAllocatedImage;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalAllocatedImage)
#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalAllocatedImage final : public MetalImage {
 public:
  LONGMARCH_RHI_QUERY(MetalAllocatedImage, MetalImage)

  static MetalAllocatedImage *Create(MetalDevice *device, const ImageDesc &desc);
  bool IsValid() const override;

 private:
  MetalAllocatedImage(MetalDevice *device, ImageDesc desc, NS::SharedPtr<MTL::Texture> texture);
  NS::SharedPtr<MTL::Texture> texture_;
};
}  // namespace grassland::rhi::backend::metal
