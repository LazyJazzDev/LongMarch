#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_sampler.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalSampler final : public Sampler {
 public:
  static MetalSampler *Create(MetalDevice *device, const SamplerDesc &desc);
  MTL::SamplerState *NativeSampler() const;

 private:
  MetalSampler(MetalDevice *device, const SamplerDesc &desc, NS::SharedPtr<MTL::SamplerState> sampler);
  NS::SharedPtr<MTL::SamplerState> sampler_;
};
}  // namespace grassland::rhi::backend::metal
