#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {
struct MetalImageState final {
  MetalImageState();
  ~MetalImageState();
  const void *device_{};
  MTL::Texture *image_{};
};
}  // namespace grassland::rhi::backend::metal
