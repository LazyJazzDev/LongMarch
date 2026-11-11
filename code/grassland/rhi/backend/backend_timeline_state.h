#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct TimelineState final {
  TimelineState();
  ~TimelineState();
  IBackendDevice *device_{};
};
}  // namespace grassland::rhi::backend
