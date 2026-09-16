#include "grassland/graphics/backend/metal/metal_image.h"

#include <cstring>
#include <stdexcept>

#include "grassland/graphics/backend/metal/metal_core.h"

namespace grassland::graphics::backend {

MetalImage::MetalImage(MetalCore *core, int width, int height, ImageFormat format)
    : core_(core), extent_{uint32_t(width), uint32_t(height)}, format_(format) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("Metal image extent must be positive");
  MetalPool pool;
  auto descriptor = MTL::TextureDescriptor::texture2DDescriptor(MetalFormat(format), width, height, false);
  // Private textures support depth targets; transfers use a shared staging buffer.
  descriptor->setStorageMode(MTL::StorageModePrivate);
  descriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget |
                       (IsDepthFormat(format) ? 0 : MTL::TextureUsageShaderWrite));
  texture_ = NS::TransferPtr(core_->Device()->newTexture(descriptor));
  MetalCheck(texture_.get(), nullptr, "newTexture");
}
namespace {
void CheckRegion(Extent2D full, Offset2D offset, Extent2D extent) {
  if (offset.x < 0 || offset.y < 0 || uint64_t(offset.x) + extent.width > full.width ||
      uint64_t(offset.y) + extent.height > full.height)
    throw std::out_of_range("Metal texture transfer");
}
}  // namespace
void MetalImage::UploadData(const void *data, const Offset2D &offset, const Extent2D &extent) const {
  CheckRegion(extent_, offset, extent);
  if (!extent.width || !extent.height)
    return;
  MetalPool pool;
  const bool rgb = format_ == IMAGE_FORMAT_R32G32B32_SFLOAT;
  const size_t pixel_size = rgb ? 16 : PixelSize(format_);
  const size_t pitch = (extent.width * pixel_size + 255) & ~size_t(255);
  auto staging = NS::TransferPtr(core_->Device()->newBuffer(pitch * extent.height, MTL::ResourceStorageModeShared));
  MetalCheck(staging.get(), nullptr, "texture upload staging");
  for (size_t y = 0; y < extent.height; ++y) {
    auto dst = static_cast<char *>(staging->contents()) + y * pitch;
    auto src = static_cast<const char *>(data) + y * extent.width * PixelSize(format_);
    if (rgb) {
      for (size_t x = 0; x < extent.width; ++x) {
        std::memcpy(dst + x * 16, src + x * 12, 12);
        float alpha = 1;
        std::memcpy(dst + x * 16 + 12, &alpha, 4);
      }
    } else
      std::memcpy(dst, src, extent.width * pixel_size);
  }
  auto command = core_->Queue()->commandBuffer();
  auto blit = command->blitCommandEncoder();
  blit->copyFromBuffer(staging.get(), 0, pitch, pitch * extent.height, MTL::Size(extent.width, extent.height, 1),
                       texture_.get(), 0, 0, MTL::Origin(offset.x, offset.y, 0));
  blit->endEncoding();
  core_->Commit(command);
  core_->WaitGPU();
}
void MetalImage::DownloadData(void *data, const Offset2D &offset, const Extent2D &extent) const {
  CheckRegion(extent_, offset, extent);
  if (!extent.width || !extent.height)
    return;
  MetalPool pool;
  const bool rgb = format_ == IMAGE_FORMAT_R32G32B32_SFLOAT;
  const size_t pixel_size = rgb ? 16 : PixelSize(format_);
  const size_t pitch = (extent.width * pixel_size + 255) & ~size_t(255);
  auto staging = NS::TransferPtr(core_->Device()->newBuffer(pitch * extent.height, MTL::ResourceStorageModeShared));
  MetalCheck(staging.get(), nullptr, "texture download staging");
  auto command = core_->Queue()->commandBuffer();
  auto blit = command->blitCommandEncoder();
  blit->copyFromTexture(texture_.get(), 0, 0, MTL::Origin(offset.x, offset.y, 0),
                        MTL::Size(extent.width, extent.height, 1), staging.get(), 0, pitch, pitch * extent.height);
  blit->endEncoding();
  core_->Commit(command);
  core_->WaitGPU();
  for (size_t y = 0; y < extent.height; ++y) {
    auto src = static_cast<const char *>(staging->contents()) + y * pitch;
    auto dst = static_cast<char *>(data) + y * extent.width * PixelSize(format_);
    if (rgb)
      for (size_t x = 0; x < extent.width; ++x)
        std::memcpy(dst + x * 12, src + x * 16, 12);
    else
      std::memcpy(dst, src, extent.width * pixel_size);
  }
}

}  // namespace grassland::graphics::backend
