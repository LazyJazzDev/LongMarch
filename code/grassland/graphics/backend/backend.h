#pragma once

#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/graphics/backend/d3d12/d3d12_backend.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/graphics/backend/vulkan/vulkan_backend.h"
#endif

namespace grassland::graphics::backend {}
