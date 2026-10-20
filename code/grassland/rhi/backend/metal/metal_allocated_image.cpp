#include "grassland/rhi/backend/metal/metal_allocated_image.h"

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
MetalAllocatedImage::MetalAllocatedImage(MetalDevice *device, ImageDesc desc, NS::SharedPtr<MTL::Texture> texture)
    : MetalImage(desc, device, texture.get()),
      texture_(std::move(texture)) {
}

MetalAllocatedImage *MetalAllocatedImage::Create(MetalDevice *device, const ImageDesc &desc) {
  const auto format = NativeImageFormat(desc.format);
  if (!desc.width || !desc.height || desc.width > 16384 || desc.height > 16384 || format == MTL::PixelFormatInvalid)
    return nullptr;
  try {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto descriptor = MTL::TextureDescriptor::texture2DDescriptor(format, desc.width, desc.height, false);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    auto usage = MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead;
    if (SupportsImageStorage(desc.format))
      usage |= MTL::TextureUsageShaderWrite;
    descriptor->setUsage(usage);
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
