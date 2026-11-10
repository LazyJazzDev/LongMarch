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
