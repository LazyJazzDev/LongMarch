#pragma once
#include "grassland/d3d12/d3d12util.h"

namespace grassland::d3d12 {
struct DeviceFeatureRequirement {
  bool enable_raytracing_extension{false};
};
}  // namespace grassland::d3d12
