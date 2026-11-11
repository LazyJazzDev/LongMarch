#pragma once
#include "grassland/rhi/backend/backend_sampler_state.h"
#include "grassland/rhi/backend/metal/metal_sampler.h"

namespace grassland::rhi::backend::metal {
class MetalSamplerImpl final : public Object<IMetalSampler> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ISampler, IMetalSampler, Sampler, ISampler)
  static IMetalSampler *Create(IMetalDevice *device, const SamplerDesc &desc);
  MTL::SamplerState *NativeSampler() const;

 private:
  MetalSamplerImpl(IMetalDevice *device, const SamplerDesc &desc, NS::SharedPtr<MTL::SamplerState> sampler);
  NS::SharedPtr<MTL::SamplerState> sampler_;

 public:
  ~MetalSamplerImpl() override {
  }

 private:
  SamplerState state_Sampler_;

  SamplerState &StateSampler() override {
    return state_Sampler_;
  }

  const SamplerState &StateSampler() const override {
    return state_Sampler_;
  }
};
}  // namespace grassland::rhi::backend::metal
