#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
enum SamplerFilter { SAMPLER_FILTER_NEAREST, SAMPLER_FILTER_LINEAR };

enum SamplerAddressMode { SAMPLER_ADDRESS_REPEAT, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MIRRORED_REPEAT };

struct SamplerDesc {
  SamplerFilter filter{SAMPLER_FILTER_LINEAR};
  SamplerAddressMode address_u{SAMPLER_ADDRESS_REPEAT}, address_v{SAMPLER_ADDRESS_REPEAT};
};

// Immutable sampler state, reusable across binding groups.
class ISampler : public IObject {
 public:
  virtual const SamplerDesc &GetDesc() const = 0;

 protected:
  ~ISampler() override = default;
};
}  // namespace grassland::rhi
