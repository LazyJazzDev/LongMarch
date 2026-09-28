#pragma once

#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {
#if defined(LONGMARCH_CUDA_RUNTIME)
VkExternalSemaphoreHandleTypeFlagBits GetDefaultExternalSemaphoreHandleType();
#endif

}  // namespace grassland::graphics::backend::vulkan
