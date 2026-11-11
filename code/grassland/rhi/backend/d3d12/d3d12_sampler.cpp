#include "grassland/rhi/backend/d3d12/d3d12_sampler.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_sampler_impl.h"

namespace grassland::rhi::backend::d3d12 {
D3D12SamplerImpl::D3D12SamplerImpl(ID3D12Device *device, const SamplerDesc &desc) {
  InitializeSampler(desc, device);
}

ID3D12Sampler *D3D12SamplerImpl::Create(ID3D12Device *device, const SamplerDesc &desc) {
  if (!Validate(desc))
    return nullptr;
  auto sampler = ObjectPtr<D3D12SamplerImpl>::Adopt(new D3D12SamplerImpl(device, desc));
  sampler->descriptor_ = device->SamplerDescriptorStorage().Acquire(1);
  if (!sampler->descriptor_)
    return nullptr;
  const D3D12_TEXTURE_ADDRESS_MODE modes[] = {D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                                              D3D12_TEXTURE_ADDRESS_MODE_MIRROR};
  auto &native = sampler->native_desc_;
  native.Filter =
      desc.filter == SAMPLER_FILTER_LINEAR ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
  native.AddressU = modes[desc.address_u];
  native.AddressV = native.AddressW = modes[desc.address_v];
  native.MaxLOD = D3D12_FLOAT32_MAX;
  native.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
  native.MaxAnisotropy = 1;
  device->NativeDevice()->CreateSampler(&native, sampler->Descriptor());
  return sampler.Detach();
}
}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12Sampler, "d1e384fa-d35c-42e0-8d5f-110c64fe99d0")

namespace grassland::rhi::backend::d3d12 {
ID3D12Sampler *ID3D12Sampler::Create(ID3D12Device *device, const SamplerDesc &desc) {
  return D3D12SamplerImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::d3d12
