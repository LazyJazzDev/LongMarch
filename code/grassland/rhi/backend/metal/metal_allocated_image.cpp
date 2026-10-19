#include "grassland/rhi/backend/metal/metal_allocated_image.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalAllocatedImage::MetalAllocatedImage(MetalDevice *device, ImageDesc desc, NS::SharedPtr<MTL::Texture> texture)
    : MetalImage(desc, device, texture.get()),
      texture_(std::move(texture)) {
}

MetalAllocatedImage *MetalAllocatedImage::Create(MetalDevice *device, const ImageDesc &desc) {
  if (!desc.width || !desc.height || desc.width > 16384 || desc.height > 16384 ||
      (desc.format != IMAGE_FORMAT_RGBA8_UNORM && desc.format != IMAGE_FORMAT_BGRA8_UNORM))
    return nullptr;
  try {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto descriptor = MTL::TextureDescriptor::texture2DDescriptor(
        desc.format == IMAGE_FORMAT_RGBA8_UNORM ? MTL::PixelFormatRGBA8Unorm : MTL::PixelFormatBGRA8Unorm, desc.width,
        desc.height, false);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage(MTL::TextureUsageRenderTarget);
    auto texture = NS::TransferPtr(device->NativeDevice()->newTexture(descriptor));
    if (!texture) {
      ReportError(-1, "Metal image allocation failed.");
      return nullptr;
    }
    return new MetalAllocatedImage(device, desc, std::move(texture));
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Metal image allocation failed.");
    return nullptr;
  }
}

bool MetalAllocatedImage::IsValid() const {
  return bool(texture_);
}
}  // namespace grassland::rhi::backend::metal
