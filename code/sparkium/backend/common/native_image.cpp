#include "sparkium/backend/common/native_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sparkium::backend {

NativeImage::NativeImage(bool cuda, int width, int height, ImageFormat format)
    : extent_{uint32_t(width), uint32_t(height)},
      format_(format),
      unorm_(false) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("native image dimensions must be positive");
  switch (format) {
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      channels_ = 4;
      unorm_ = true;
      break;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      channels_ = 4;
      break;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      channels_ = 3;
      break;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      channels_ = 2;
      break;
    case IMAGE_FORMAT_R32_SFLOAT:
    case IMAGE_FORMAT_D32_SFLOAT:
    case IMAGE_FORMAT_R32_UINT:
    case IMAGE_FORMAT_R32_SINT:
      channels_ = 1;
      break;
    default:
      throw std::invalid_argument("unsupported native image format");
  }
  external_bytes_ = channels_ * (unorm_ ? 1 : 4);
  if (uint64_t(width) * height > std::numeric_limits<size_t>::max() / (channels_ * 4))
    throw std::overflow_error("native image size overflow");
  // Keep SDR texels packed. Expanding all imported textures to float4 wastes
  // four times their memory and prevents large pinned scenes fitting on CPU.
  memory = std::make_unique<NativeMemory>(cuda, size_t(width) * height * external_bytes_);
}

namespace {
void CheckRegion(Extent2D full, Offset2D o, Extent2D e) {
  if (o.x < 0 || o.y < 0 || uint64_t(o.x) + e.width > full.width || uint64_t(o.y) + e.height > full.height)
    throw std::out_of_range("native image region");
}
}  // namespace

void NativeImage::UploadData(const void *data, const Offset2D &o, const Extent2D &e) const {
  CheckRegion(extent_, o, e);
  std::vector<uint8_t> row(size_t(e.width) * external_bytes_);
  for (uint32_t y = 0; y < e.height; ++y) {
    const auto *src = static_cast<const uint8_t *>(data) + size_t(y) * e.width * external_bytes_;
    std::memcpy(row.data(), src, row.size());
    if (format_ == IMAGE_FORMAT_B8G8R8A8_UNORM)
      for (uint32_t x = 0; x < e.width; ++x)
        std::swap(row[x * 4], row[x * 4 + 2]);
    memory->Upload(row.data(), row.size(), (size_t(y + o.y) * extent_.width + o.x) * external_bytes_);
  }
}

void NativeImage::DownloadData(void *data, const Offset2D &o, const Extent2D &e) const {
  CheckRegion(extent_, o, e);
  std::vector<uint8_t> row(size_t(e.width) * external_bytes_);
  for (uint32_t y = 0; y < e.height; ++y) {
    memory->Download(row.data(), row.size(), (size_t(y + o.y) * extent_.width + o.x) * external_bytes_);
    auto *dst = static_cast<uint8_t *>(data) + size_t(y) * e.width * external_bytes_;
    if (format_ == IMAGE_FORMAT_B8G8R8A8_UNORM)
      for (uint32_t x = 0; x < e.width; ++x)
        std::swap(row[x * 4], row[x * 4 + 2]);
    std::memcpy(dst, row.data(), row.size());
  }
}

void NativeImage::Clear(const ClearValue &value) {
  if (unorm_) {
    const float c[]{value.color.r, value.color.g, value.color.b, value.color.a};
    std::vector<uint8_t> pixels(size_t(extent_.width) * extent_.height * 4);
    for (size_t i = 0; i < pixels.size(); ++i)
      pixels[i] = uint8_t(std::nearbyint(std::clamp(c[i % 4], 0.0f, 1.0f) * 255.0f));
    memory->Upload(pixels.data(), pixels.size());
    return;
  }

  std::vector<float> pixels(size_t(extent_.width) * extent_.height * channels_);
  const float c[4] = {value.color.r, value.color.g, value.color.b, value.color.a};
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = c[i % channels_];
  memory->Upload(pixels.data(), pixels.size() * 4);
}

}  // namespace sparkium::backend
