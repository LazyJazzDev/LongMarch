#include "grassland/rhi/backend/d3d12/d3d12_sampler.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12SamplerImpl::D3D12SamplerImpl(ID3D12Device *device, const SamplerDesc &desc) {
  InitializeSampler(desc, device);
}

rhi::ISampler *D3D12SamplerImpl::Create(ID3D12Device *device, const SamplerDesc &desc) {
  if (!ValidateSamplerFeatures(desc, device->GetPhysicalDevice().mip_feature))
    return nullptr;
  auto sampler = ObjectPtr<D3D12SamplerImpl>::Adopt(new D3D12SamplerImpl(device, desc));
  sampler->descriptor_ = device->SamplerDescriptorStorage().Acquire(1);
  if (!sampler->descriptor_)
    return nullptr;
  const D3D12_TEXTURE_ADDRESS_MODE modes[] = {D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                                              D3D12_TEXTURE_ADDRESS_MODE_MIRROR, D3D12_TEXTURE_ADDRESS_MODE_BORDER};
  auto &native = sampler->native_desc_;
  native.Filter = desc.max_anisotropy > 1 ? D3D12_FILTER_ANISOTROPIC
                                          : D3D12_ENCODE_BASIC_FILTER(static_cast<D3D12_FILTER_TYPE>(desc.MinFilter()),
                                                                      static_cast<D3D12_FILTER_TYPE>(desc.MagFilter()),
                                                                      static_cast<D3D12_FILTER_TYPE>(desc.MipFilter()),
                                                                      D3D12_FILTER_REDUCTION_TYPE_STANDARD);
  native.AddressU = modes[desc.address_u];
  native.AddressV = modes[desc.address_v];
  native.AddressW = modes[desc.address_w];
  const float colors[][4] = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}};
  for (int i = 0; i < 4; ++i)
    native.BorderColor[i] = colors[desc.border_color][i];
  native.MinLOD = desc.min_lod;
  native.MaxLOD = desc.max_lod;
  native.MipLODBias = desc.mip_lod_bias;
  native.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
  native.MaxAnisotropy = desc.max_anisotropy;
  device->NativeDevice()->CreateSampler(&native, sampler->Descriptor());
  return sampler.Detach();
}

rhi::ISampler *ID3D12Sampler::Create(ID3D12Device *device, const SamplerDesc &desc) {
  return D3D12SamplerImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::d3d12
