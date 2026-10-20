#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend::metal {
MTL::PixelFormat NativeImageFormat(ImageFormat format);
}
