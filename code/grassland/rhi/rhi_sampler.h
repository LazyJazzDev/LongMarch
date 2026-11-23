#pragma once
#include <limits>

#include "grassland/rhi/rhi_object.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::ISampler, grassland::rhi::IObject, "1686d5c9-e228-4896-a8db-fd1ee89e98d2")

namespace grassland::rhi {
enum SamplerFilter { SAMPLER_FILTER_INHERIT = -1, SAMPLER_FILTER_NEAREST, SAMPLER_FILTER_LINEAR };

enum SamplerAddressMode { SAMPLER_ADDRESS_REPEAT, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MIRRORED_REPEAT };

struct SamplerDesc {
  SamplerDesc() = default;
  // Legacy construction and writes to filter remain supported. Independent
  // filters override it when they are not INHERIT.
  SamplerDesc(SamplerFilter filter,
              SamplerAddressMode address_u = SAMPLER_ADDRESS_REPEAT,
              SamplerAddressMode address_v = SAMPLER_ADDRESS_REPEAT);
  SamplerDesc(SamplerFilter min_filter,
              SamplerFilter mag_filter,
              SamplerFilter mip_filter,
              SamplerAddressMode address_u = SAMPLER_ADDRESS_REPEAT,
              SamplerAddressMode address_v = SAMPLER_ADDRESS_REPEAT,
              SamplerAddressMode address_w = SAMPLER_ADDRESS_REPEAT);

  SamplerFilter filter{SAMPLER_FILTER_LINEAR};
  SamplerAddressMode address_u{SAMPLER_ADDRESS_REPEAT}, address_v{SAMPLER_ADDRESS_REPEAT};
  SamplerFilter min_filter{SAMPLER_FILTER_INHERIT};
  SamplerFilter mag_filter{SAMPLER_FILTER_INHERIT};
  SamplerFilter mip_filter{SAMPLER_FILTER_INHERIT};
  SamplerAddressMode address_w{SAMPLER_ADDRESS_REPEAT};
  float min_lod{0.0f};
  float max_lod{(std::numeric_limits<float>::max)()};
  float mip_lod_bias{0.0f};
  // 1 disables anisotropic filtering. Larger values require linear filters.
  uint32_t max_anisotropy{1};

  SamplerFilter MinFilter() const;
  SamplerFilter MagFilter() const;
  SamplerFilter MipFilter() const;
};

// Immutable sampler state, reusable across binding groups.
class ISampler : public IObject {
 public:
  virtual const SamplerDesc &GetDesc() const = 0;

 protected:
  ~ISampler() override = default;
};
}  // namespace grassland::rhi
