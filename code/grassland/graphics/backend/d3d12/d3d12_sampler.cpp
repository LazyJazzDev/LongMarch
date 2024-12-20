#include "grassland/graphics/backend/d3d12/d3d12_sampler.h"

namespace grassland::graphics::backend {

D3D12Sampler::D3D12Sampler(D3D12Core *core, const SamplerInfo &info)
    : core_(core) {
  sampler_desc_.Filter = FilterModeToD3D12Filter(
      info.min_filter, info.mag_filter, info.mip_filter);
  sampler_desc_.AddressU = AddressModeToD3D12AddressMode(info.address_mode_u);
  sampler_desc_.AddressV = AddressModeToD3D12AddressMode(info.address_mode_v);
  sampler_desc_.AddressW = AddressModeToD3D12AddressMode(info.address_mode_w);
  sampler_desc_.MipLODBias = 0.0f;
  sampler_desc_.MaxAnisotropy = 1;
  sampler_desc_.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
  sampler_desc_.BorderColor[0] = 0.0f;
  sampler_desc_.BorderColor[1] = 0.0f;
  sampler_desc_.BorderColor[2] = 0.0f;
  sampler_desc_.BorderColor[3] = 0.0f;
  sampler_desc_.MinLOD = 0.0f;
  sampler_desc_.MaxLOD = D3D12_FLOAT32_MAX;
}

}  // namespace grassland::graphics::backend
