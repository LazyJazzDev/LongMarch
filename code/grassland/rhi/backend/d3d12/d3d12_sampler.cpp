#include "grassland/rhi/backend/d3d12/d3d12_sampler.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12SamplerImpl::D3D12SamplerImpl(ID3D12Device *device, const SamplerDesc &desc) {
  InitializeSampler(desc, device);
}

rhi::ISampler *D3D12SamplerImpl::Create(ID3D12Device *device, const SamplerDesc &desc) {
  if (!ValidateSamplerDesc(desc))
    return nullptr;
  auto sampler = ObjectPtr<D3D12SamplerImpl>::Adopt(new D3D12SamplerImpl(device, desc));
  sampler->descriptor_ = device->SamplerDescriptorStorage().Acquire(1);
  if (!sampler->descriptor_)
    return nullptr;
  const D3D12_TEXTURE_ADDRESS_MODE modes[] = {D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                                              D3D12_TEXTURE_ADDRESS_MODE_MIRROR};
  auto &native = sampler->native_desc_;
  native.Filter =
      desc.max_anisotropy > 1
          ? (desc.comparison ? D3D12_FILTER_COMPARISON_ANISOTROPIC : D3D12_FILTER_ANISOTROPIC)
          : D3D12_ENCODE_BASIC_FILTER(
                desc.filter == SAMPLER_FILTER_LINEAR ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
                desc.filter == SAMPLER_FILTER_LINEAR ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
                desc.mip_filter == SAMPLER_FILTER_LINEAR ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
                desc.comparison ? D3D12_FILTER_REDUCTION_TYPE_COMPARISON : D3D12_FILTER_REDUCTION_TYPE_STANDARD);
  native.AddressU = modes[desc.address_u];
  native.AddressV = native.AddressW = modes[desc.address_v];
  native.MinLOD = desc.min_lod;
  native.MaxLOD = desc.max_lod;
  native.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(desc.comparison.value_or(COMPARE_ALWAYS) + 1);
  native.MaxAnisotropy = desc.max_anisotropy;
  device->NativeDevice()->CreateSampler(&native, sampler->Descriptor());
  return sampler.Detach();
}

rhi::ISampler *ID3D12Sampler::Create(ID3D12Device *device, const SamplerDesc &desc) {
  return D3D12SamplerImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::d3d12
