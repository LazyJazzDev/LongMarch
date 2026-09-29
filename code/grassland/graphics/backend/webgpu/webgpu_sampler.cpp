#include "grassland/graphics/backend/webgpu/webgpu_sampler.h"

#include "grassland/graphics/backend/webgpu/webgpu_core.h"

namespace grassland::graphics::backend {

namespace {
wgpu::AddressMode Address(AddressMode mode) {
  switch (mode) {
    case ADDRESS_MODE_REPEAT:
      return wgpu::AddressMode::Repeat;
    case ADDRESS_MODE_MIRRORED_REPEAT:
      return wgpu::AddressMode::MirrorRepeat;
    default:
      // WebGPU has no border colors; clamping is the closest mode.
      return wgpu::AddressMode::ClampToEdge;
  }
}

wgpu::FilterMode Filter(FilterMode mode) {
  return mode == FILTER_MODE_LINEAR ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
}
}  // namespace

WebGPUSampler::WebGPUSampler(WebGPUCore *core, const SamplerInfo &info) {
  wgpu::SamplerDescriptor descriptor{};
  descriptor.addressModeU = Address(info.address_mode_u);
  descriptor.addressModeV = Address(info.address_mode_v);
  descriptor.addressModeW = Address(info.address_mode_w);
  descriptor.minFilter = Filter(info.min_filter);
  descriptor.magFilter = Filter(info.mag_filter);
  descriptor.mipmapFilter =
      info.mip_filter == FILTER_MODE_LINEAR ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest;
  sampler_ = core->Device().CreateSampler(&descriptor);
}

}  // namespace grassland::graphics::backend
