#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalSampler,
                        grassland::rhi::backend::IBackendSampler,
                        "c609fe39-6f33-4a10-a4b1-db4fa86332b7")

namespace grassland::rhi::backend::metal {
class IMetalSampler : public IBackendSampler {
 public:
  static rhi::ISampler *Create(IMetalDevice *device, const SamplerDesc &desc);
  virtual MTL::SamplerState *NativeSampler() const = 0;
};

}  // namespace grassland::rhi::backend::metal
