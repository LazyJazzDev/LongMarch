#include "grassland/rhi/backend/metal/metal_sampler.h"

#include "grassland/rhi/backend/metal/metal_device.h"

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
  // Metal has no sampler-level LOD bias; do not silently ignore it.
  if (!ValidateSamplerDesc(desc) || desc.max_anisotropy > 16 || desc.mip_lod_bias != 0.0f)
    return nullptr;

  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto descriptor = NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init());
  descriptor->setSupportArgumentBuffers(true);
  descriptor->setMinFilter(desc.MinFilter() == SAMPLER_FILTER_NEAREST ? MTL::SamplerMinMagFilterNearest
                                                                      : MTL::SamplerMinMagFilterLinear);
  descriptor->setMagFilter(desc.MagFilter() == SAMPLER_FILTER_NEAREST ? MTL::SamplerMinMagFilterNearest
                                                                      : MTL::SamplerMinMagFilterLinear);
  descriptor->setMipFilter(desc.MipFilter() == SAMPLER_FILTER_NEAREST ? MTL::SamplerMipFilterNearest
                                                                      : MTL::SamplerMipFilterLinear);
  descriptor->setLodMinClamp(desc.min_lod);
  descriptor->setLodMaxClamp(desc.max_lod);
  descriptor->setMaxAnisotropy(desc.max_anisotropy);
  const MTL::SamplerAddressMode modes[] = {MTL::SamplerAddressModeRepeat, MTL::SamplerAddressModeClampToEdge,
                                           MTL::SamplerAddressModeMirrorRepeat};
  descriptor->setSAddressMode(modes[desc.address_u]);
  descriptor->setTAddressMode(modes[desc.address_v]);
  descriptor->setRAddressMode(modes[desc.address_w]);
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
