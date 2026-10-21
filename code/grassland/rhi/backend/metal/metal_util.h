#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend::metal {
bool SupportsImageStorage(ImageFormat format);
MTL::PixelFormat NativeImageFormat(ImageFormat format);
}  // namespace grassland::rhi::backend::metal
