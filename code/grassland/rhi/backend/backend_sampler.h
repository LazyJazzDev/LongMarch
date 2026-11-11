#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendSampler)
#include "grassland/rhi/rhi_sampler.h"

namespace grassland::rhi::backend {
struct SamplerState;

class IBackendSampler : public ISampler {
 protected:
  void InitializeSampler(const SamplerDesc &desc, const void *device);

 public:
  const SamplerDesc &GetDesc() const override = 0;
  const SamplerDesc &CommonGetDesc() const;
  const void *DeviceIdentity() const;
  static bool Validate(const SamplerDesc &desc);

 protected:
  virtual SamplerState &StateSampler() = 0;
  virtual const SamplerState &StateSampler() const = 0;
};
}  // namespace grassland::rhi::backend
