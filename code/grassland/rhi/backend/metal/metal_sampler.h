#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class IMetalSampler;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalSampler)
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_sampler.h"

namespace grassland::rhi::backend::metal {
class IMetalDevice;

class IMetalSampler : public Sampler {
 public:
  static IMetalSampler *Create(IMetalDevice *device, const SamplerDesc &desc);
  virtual MTL::SamplerState *NativeSampler() const = 0;
};

;
}  // namespace grassland::rhi::backend::metal
