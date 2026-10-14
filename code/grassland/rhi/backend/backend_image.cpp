#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {

Image::Image(ImageDesc desc) : desc_(desc) {
}

const ImageDesc &Image::GetDesc() const {
  return desc_;
}

bool Image::Claim() {
  bool expected = false;
  return claimed_.compare_exchange_strong(expected, true);
}

void Image::CancelClaim() {
  claimed_ = false;
}

}  // namespace grassland::rhi::backend
