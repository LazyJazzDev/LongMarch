#include "grassland/rhi/rhi_sampler.h"

namespace grassland::rhi {
SamplerDesc::SamplerDesc(SamplerFilter value, SamplerAddressMode u, SamplerAddressMode v)
    : filter(value),
      address_u(u),
      address_v(v),
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
  return min_filter == SAMPLER_FILTER_INHERIT ? filter : min_filter;
}

SamplerFilter SamplerDesc::MagFilter() const {
  return mag_filter == SAMPLER_FILTER_INHERIT ? filter : mag_filter;
}

SamplerFilter SamplerDesc::MipFilter() const {
  return mip_filter == SAMPLER_FILTER_INHERIT ? filter : mip_filter;
}
}  // namespace grassland::rhi
