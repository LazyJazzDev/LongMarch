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
  const auto &limits = device->GetPhysicalDevice().mip_feature;
  if (!ValidateSamplerDesc(desc) || desc.max_anisotropy > limits.max_anisotropy ||
      desc.mip_lod_bias < limits.min_lod_bias || desc.mip_lod_bias > limits.max_lod_bias)
    return nullptr;
  if (desc.compare_enable && !device->NativeDevice()->supportsFamily(MTL::GPUFamilyApple3) &&
      !device->NativeDevice()->supportsFamily(MTL::GPUFamilyMac2)) {
    ReportError(-1, "Metal comparison sampling requires Apple3 or Mac2 GPU support.");
    return nullptr;
  }
  const bool uses_border = desc.address_u == SAMPLER_ADDRESS_CLAMP_TO_BORDER ||
                           desc.address_v == SAMPLER_ADDRESS_CLAMP_TO_BORDER ||
                           desc.address_w == SAMPLER_ADDRESS_CLAMP_TO_BORDER;
  if (uses_border && !device->NativeDevice()->supportsFamily(MTL::GPUFamilyApple7) &&
      !device->NativeDevice()->supportsFamily(MTL::GPUFamilyMac2)) {
    ReportError(-1, "Metal border-color sampling requires Apple7 or Mac2 GPU support.");
    return nullptr;
  }

  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto descriptor = NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init());
  descriptor->setSupportArgumentBuffers(true);
  descriptor->setCompareFunction(desc.compare_enable ? static_cast<MTL::CompareFunction>(desc.compare_op)
                                                     : MTL::CompareFunctionNever);
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
                                           MTL::SamplerAddressModeMirrorRepeat,
                                           MTL::SamplerAddressModeClampToBorderColor};
  descriptor->setSAddressMode(modes[desc.address_u]);
  descriptor->setTAddressMode(modes[desc.address_v]);
  descriptor->setRAddressMode(modes[desc.address_w]);
  const MTL::SamplerBorderColor colors[] = {MTL::SamplerBorderColorTransparentBlack, MTL::SamplerBorderColorOpaqueBlack,
                                            MTL::SamplerBorderColorOpaqueWhite};
  if (uses_border)
    descriptor->setBorderColor(colors[desc.border_color]);
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
