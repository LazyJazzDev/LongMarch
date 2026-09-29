#include "grassland/graphics/backend/webgpu/webgpu_image.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "grassland/graphics/backend/webgpu/webgpu_core.h"

namespace grassland::graphics::backend {

namespace {
bool Storable(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
    case IMAGE_FORMAT_R32G32_SFLOAT:
    case IMAGE_FORMAT_R32_SFLOAT:
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
    case IMAGE_FORMAT_R32_UINT:
    case IMAGE_FORMAT_R32_SINT:
      return true;
    default:
      return false;
  }
}

void CheckRegion(Extent2D full, Offset2D offset, Extent2D extent) {
  if (offset.x < 0 || offset.y < 0 || uint64_t(offset.x) + extent.width > full.width ||
      uint64_t(offset.y) + extent.height > full.height)
    throw std::out_of_range("WebGPU texture transfer");
}
}  // namespace

WebGPUImage::WebGPUImage(WebGPUCore *core, int width, int height, ImageFormat format)
    : core_(core),
      extent_{uint32_t(width), uint32_t(height)},
      format_(format) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("WebGPU image extent must be positive");
  wgpu::TextureDescriptor descriptor{};
  descriptor.dimension = wgpu::TextureDimension::e2D;
  descriptor.size = {uint32_t(width), uint32_t(height), 1};
  descriptor.format = WebGPUFormat(format);
  descriptor.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment |
                     wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc;
  if (Storable(format))
    descriptor.usage |= wgpu::TextureUsage::StorageBinding;
  texture_ = core_->Device().CreateTexture(&descriptor);
  view_ = texture_.CreateView();
}

void WebGPUImage::UploadData(const void *data, const Offset2D &offset, const Extent2D &extent) const {
  CheckRegion(extent_, offset, extent);
  if (!extent.width || !extent.height)
    return;
  const size_t texel = WebGPUTexelSize(format_), source_texel = PixelSize(format_);
  std::vector<uint8_t> widened;
  const void *bytes = data;
  if (texel != source_texel) {
    // RGB32F is stored as RGBA32F with an opaque alpha, like the Metal backend.
    widened.resize(size_t(extent.width) * extent.height * texel);
    const float alpha = 1;
    for (size_t i = 0; i < size_t(extent.width) * extent.height; ++i) {
      std::memcpy(widened.data() + i * texel, static_cast<const char *>(data) + i * source_texel, source_texel);
      std::memcpy(widened.data() + i * texel + 12, &alpha, 4);
    }
    bytes = widened.data();
  }
  wgpu::TexelCopyTextureInfo destination{};
  destination.texture = texture_;
  destination.origin = {uint32_t(offset.x), uint32_t(offset.y), 0};
  wgpu::TexelCopyBufferLayout layout{};
  layout.bytesPerRow = uint32_t(extent.width * texel);
  layout.rowsPerImage = extent.height;
  wgpu::Extent3D size{extent.width, extent.height, 1};
  core_->Queue().WriteTexture(&destination, bytes, size_t(layout.bytesPerRow) * extent.height, &layout, &size);
}

void WebGPUImage::DownloadData(void *, const Offset2D &, const Extent2D &) const {
  throw std::runtime_error("WebGPU texture readback is asynchronous and unavailable here");
}

}  // namespace grassland::graphics::backend
