#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalSampler)

namespace grassland::rhi::backend::metal {
class IMetalSampler : public IBackendSampler {
 public:
  static IMetalSampler *Create(IMetalDevice *device, const SamplerDesc &desc);
  virtual MTL::SamplerState *NativeSampler() const = 0;
};

}  // namespace grassland::rhi::backend::metal
