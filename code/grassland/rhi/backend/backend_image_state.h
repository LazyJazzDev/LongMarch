#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct ImageState final {
  ImageState();
  ~ImageState();
  ImageDesc desc_{};
  IBackendQueue *queue_{};
};
}  // namespace grassland::rhi::backend
