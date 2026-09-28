#pragma once
#include "grassland/graphics/backend/d3d12/helper/d3d12util.h"

namespace grassland::graphics::backend::d3d12 {
struct DeviceFeatureRequirement {
  bool enable_raytracing_extension{false};
};
}  // namespace grassland::graphics::backend::d3d12
