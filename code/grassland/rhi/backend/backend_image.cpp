#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {

Image::Image(ImageDesc desc) : desc_(desc) {
}

const ImageDesc &Image::GetDesc() const {
  return desc_;
}

}  // namespace grassland::rhi::backend
