#pragma once
#include "cao_di/d3d12/d3d12util.h"

namespace CD::d3d12 {
struct DeviceFeatureRequirement {
  bool enable_raytracing_extension{false};
};
}  // namespace CD::d3d12
