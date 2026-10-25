#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend::metal {
MTL::PrimitiveType NativePrimitiveTopology(PrimitiveTopology topology);

MTL::BlendFactor NativeBlendFactor(BlendFactor factor, bool alpha = false);
MTL::BlendOperation NativeBlendOp(BlendOp op);
bool SupportsImageStorage(ImageFormat format);
MTL::PixelFormat NativeImageFormat(ImageFormat format);
}  // namespace grassland::rhi::backend::metal
