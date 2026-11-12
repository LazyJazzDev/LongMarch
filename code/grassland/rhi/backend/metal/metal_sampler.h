#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalSampler,
                        grassland::rhi::IObject,
                        "c609fe39-6f33-4a10-a4b1-db4fa86332b7")

namespace grassland::rhi::backend::metal {
class IMetalSampler : public IObject {
 public:
  virtual const void *DeviceIdentity() const = 0;
  static rhi::ISampler *Create(IMetalDevice *device, const SamplerDesc &desc);
  virtual MTL::SamplerState *NativeSampler() const = 0;
};

}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
class MetalSamplerImpl final : public Object<rhi::ISampler, IMetalSampler> {
 public:
  static rhi::ISampler *Create(IMetalDevice *device, const SamplerDesc &desc);

  MTL::SamplerState *NativeSampler() const;

  ~MetalSamplerImpl() override = default;

  const SamplerDesc &GetDesc() const override {
    return desc_;
  }

  const void *DeviceIdentity() const override {
    return device_identity_;
  }

 private:
  MetalSamplerImpl(IMetalDevice *device, const SamplerDesc &desc, NS::SharedPtr<MTL::SamplerState> sampler);

  NS::SharedPtr<MTL::SamplerState> sampler_;

  void InitializeSampler(const SamplerDesc &desc, const void *device) {
    desc_ = desc;
    device_identity_ = device;
  }

  SamplerDesc desc_{};

  const void *device_identity_{};
};
}  // namespace grassland::rhi::backend::metal
