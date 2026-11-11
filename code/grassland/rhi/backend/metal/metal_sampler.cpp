#include "grassland/rhi/backend/metal/metal_sampler.h"

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_sampler_impl.h"

namespace grassland::rhi::backend::metal {
MetalSamplerImpl::MetalSamplerImpl(IMetalDevice *device,
                                   const SamplerDesc &desc,
                                   NS::SharedPtr<MTL::SamplerState> sampler)
    : sampler_(std::move(sampler)) {
  InitializeSampler(desc, device);
}

MTL::SamplerState *MetalSamplerImpl::NativeSampler() const {
  return sampler_.get();
}

rhi::ISampler *MetalSamplerImpl::Create(IMetalDevice *device, const SamplerDesc &desc) {
  if (!Validate(desc))
    return nullptr;

  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto descriptor = NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init());
  descriptor->setSupportArgumentBuffers(true);
  descriptor->setMinFilter(desc.filter == SAMPLER_FILTER_NEAREST ? MTL::SamplerMinMagFilterNearest
                                                                 : MTL::SamplerMinMagFilterLinear);
  descriptor->setMagFilter(descriptor->minFilter());
  const MTL::SamplerAddressMode modes[] = {MTL::SamplerAddressModeRepeat, MTL::SamplerAddressModeClampToEdge,
                                           MTL::SamplerAddressModeMirrorRepeat};
  descriptor->setSAddressMode(modes[desc.address_u]);
  descriptor->setTAddressMode(modes[desc.address_v]);
  descriptor->setRAddressMode(modes[desc.address_v]);
  auto sampler = NS::TransferPtr(device->NativeDevice()->newSamplerState(descriptor.get()));
  if (!sampler) {
    ReportError(-1, "Metal sampler creation failed.");
    return nullptr;
  }
  return new MetalSamplerImpl(device, desc, std::move(sampler));
}

rhi::ISampler *IMetalSampler::Create(IMetalDevice *device, const SamplerDesc &desc) {
  return MetalSamplerImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::metal
