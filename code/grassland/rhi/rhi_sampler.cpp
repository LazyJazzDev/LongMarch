#include "grassland/rhi/rhi_sampler.h"

namespace grassland::rhi {
SamplerDesc::SamplerDesc(SamplerFilter value, SamplerAddressMode u, SamplerAddressMode v)
    : address_u(u),
      address_v(v),
      min_filter(value),
      mag_filter(value),
      mip_filter(value),
      address_w(v) {
}

SamplerDesc::SamplerDesc(SamplerFilter min,
                         SamplerFilter mag,
                         SamplerFilter mip,
                         SamplerAddressMode u,
                         SamplerAddressMode v,
                         SamplerAddressMode w)
    : address_u(u),
      address_v(v),
      min_filter(min),
      mag_filter(mag),
      mip_filter(mip),
      address_w(w) {
}

SamplerFilter SamplerDesc::MinFilter() const {
  return min_filter;
}

SamplerFilter SamplerDesc::MagFilter() const {
  return mag_filter;
}

SamplerFilter SamplerDesc::MipFilter() const {
  return mip_filter;
}
}  // namespace grassland::rhi
