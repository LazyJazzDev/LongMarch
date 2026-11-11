#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_sampler.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendSampler,
                        grassland::rhi::IObject,
                        "34cc865a-50d9-49f8-b8db-591bc452621f")

namespace grassland::rhi::backend {
class IBackendSampler : public IObject {
 public:
  virtual const void *DeviceIdentity() const = 0;
  static bool Validate(const SamplerDesc &desc);
};
}  // namespace grassland::rhi::backend
