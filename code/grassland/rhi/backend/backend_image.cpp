#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {

IImage::IImage(ImageDesc desc) : desc_(desc) {
}

const ImageDesc &IImage::GetDesc() const {
  return desc_;
}

bool IImage::Claim() {
  bool expected = false;
  return claimed_.compare_exchange_strong(expected, true);
}

void IImage::CancelClaim() {
  claimed_ = false;
}

}  // namespace grassland::rhi::backend
