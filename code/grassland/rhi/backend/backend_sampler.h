#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Sampler;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Sampler)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_sampler.h"

namespace grassland::rhi::backend {
class Sampler : public Object<ISampler> {
 public:
  LONGMARCH_RHI_QUERY(Sampler, Object<ISampler>)

  Sampler(const SamplerDesc &desc, const void *device) : desc_(desc), device_(device) {
  }

  const SamplerDesc &GetDesc() const override {
    return desc_;
  }

  const void *DeviceIdentity() const {
    return device_;
  }

  static bool Validate(const SamplerDesc &desc);

 private:
  SamplerDesc desc_;
  const void *device_;
};
}  // namespace grassland::rhi::backend
