#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Sampler;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Sampler)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_sampler.h"

namespace grassland::rhi::backend {
struct SamplerState;

class Sampler : public ISampler {
 protected:
  void InitializeSampler(const SamplerDesc &desc, const void *device);

 public:
  const SamplerDesc &GetDesc() const override;
  const void *DeviceIdentity() const;
  static bool Validate(const SamplerDesc &desc);

 protected:
  virtual SamplerState &StateSampler() = 0;
  virtual const SamplerState &StateSampler() const = 0;
};
}  // namespace grassland::rhi::backend
