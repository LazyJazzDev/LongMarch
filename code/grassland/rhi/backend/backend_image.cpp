#include "grassland/rhi/backend/backend_image.h"

#include "grassland/rhi/backend/backend_image_state.h"

namespace grassland::rhi::backend {

void Image::InitializeImage(ImageDesc desc, Queue *queue) {
  StateImage().desc_ = desc;
  StateImage().queue_ = queue;
}

bool Image::SupportsQueue(const Queue *queue) const {
  return !StateImage().queue_ || StateImage().queue_ == queue;
}

bool Image::SupportsStorage() const {
  return false;
}

const ImageDesc &Image::GetDesc() const {
  return StateImage().desc_;
}

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::Image, "1c62d59a-7c6e-4442-8855-d6b1d1b4804c")

namespace grassland::rhi::backend {
ImageState::ImageState() = default;
ImageState::~ImageState() = default;
}  // namespace grassland::rhi::backend
