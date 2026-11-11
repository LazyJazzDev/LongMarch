#pragma once
#include "grassland/rhi/backend/backend_timeline.h"

namespace grassland::rhi::backend {
struct TimelineState final {
  TimelineState();
  ~TimelineState();
  Device *device_{};
};
}  // namespace grassland::rhi::backend
