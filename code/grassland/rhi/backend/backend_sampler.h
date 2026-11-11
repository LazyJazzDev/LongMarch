#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_sampler.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendSampler)

namespace grassland::rhi::backend {
class IBackendSampler : public IObject {
 public:
  virtual const void *DeviceIdentity() const = 0;
  static bool Validate(const SamplerDesc &desc);
};
}  // namespace grassland::rhi::backend
