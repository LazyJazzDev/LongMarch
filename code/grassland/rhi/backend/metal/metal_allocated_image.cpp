#include "grassland/rhi/backend/metal/metal_allocated_image.h"

#include "grassland/rhi/backend/metal/metal_allocated_image_impl.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
MetalAllocatedImageImpl::MetalAllocatedImageImpl(IMetalDevice *device,
                                                 ImageDesc desc,
                                                 NS::SharedPtr<MTL::Texture> texture)
    : texture_(std::move(texture)) {
  InitializeMetalImage(desc, device, texture_.get());
}

IImage *MetalAllocatedImageImpl::Create(IMetalDevice *device, const ImageDesc &desc) {
  const auto format = NativeImageFormat(desc.format);
  if (!desc.width || !desc.height || desc.width > 16384 || desc.height > 16384 || format == MTL::PixelFormatInvalid)
    return nullptr;

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
  return new MetalAllocatedImageImpl(device, desc, std::move(texture));
}

bool MetalAllocatedImageImpl::IsValid() const {
  return bool(texture_);
}

IImage *CreateMetalAllocatedImage(IMetalDevice *device, const ImageDesc &desc) {
  return MetalAllocatedImageImpl::Create(device, desc);
}

void MetalAllocatedImageImpl::InitializeImage(ImageDesc desc, IBackendQueue *queue) {
  desc_ = desc;
  queue_ = queue;
}

bool MetalAllocatedImageImpl::SupportsQueue(const IBackendQueue *queue) const {
  return !queue_ || queue_ == queue;
}

const ImageDesc &MetalAllocatedImageImpl::GetDesc() const {
  return desc_;
}

void MetalAllocatedImageImpl::InitializeMetalImage(ImageDesc desc,
                                                   const void *device,
                                                   MTL::Texture *image,
                                                   IBackendQueue *queue) {
  InitializeImage(desc, queue);
  device_ = device;
}

bool MetalAllocatedImageImpl::SupportsStorage() const {
  return (texture_->usage() & MTL::TextureUsageShaderWrite) != 0;
}

const void *MetalAllocatedImageImpl::DeviceIdentity() const {
  return device_;
}

MTL::Texture *MetalAllocatedImageImpl::NativeImage() const {
  return texture_.get();
}
}  // namespace grassland::rhi::backend::metal
