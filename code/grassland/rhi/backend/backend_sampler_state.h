#pragma once
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct SamplerState final {
  SamplerState();
  ~SamplerState();
  SamplerDesc desc_{};
  const void *device_{};
};
}  // namespace grassland::rhi::backend
