#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {

Image::Image(ImageDesc desc, Queue *queue) : desc_(desc), queue_(queue) {
}

bool Image::SupportsQueue(const Queue *queue) const {
  return !queue_ || queue_ == queue;
}

bool Image::SupportsStorage() const {
  return false;
}

const ImageDesc &Image::GetDesc() const {
  return desc_;
}

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::Image, "1c62d59a-7c6e-4442-8855-d6b1d1b4804c")
