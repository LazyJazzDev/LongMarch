#pragma once
#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {
struct ImageState final {
  ImageState();
  ~ImageState();
  ImageDesc desc_{};
  Queue *queue_{};
};
}  // namespace grassland::rhi::backend
