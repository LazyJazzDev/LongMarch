#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {

Image::Image(ImageDesc desc) : desc_(desc) {
}

bool Image::SupportsQueue(const Queue *) const {
  return true;
}

bool Image::SupportsStorage() const {
  return false;
}

const ImageDesc &Image::GetDesc() const {
  return desc_;
}

}  // namespace grassland::rhi::backend
