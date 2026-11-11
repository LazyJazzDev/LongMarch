#pragma once
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
IImage *CreateMetalAllocatedImage(IMetalDevice *device, const ImageDesc &desc);

}  // namespace grassland::rhi::backend::metal
