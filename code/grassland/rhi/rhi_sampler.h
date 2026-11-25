#pragma once
#include <limits>

#include "grassland/rhi/rhi_object.h"
#include "grassland/rhi/rhi_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::ISampler, grassland::rhi::IObject, "1686d5c9-e228-4896-a8db-fd1ee89e98d2")

namespace grassland::rhi {
enum SamplerFilter { SAMPLER_FILTER_NEAREST, SAMPLER_FILTER_LINEAR };

enum SamplerAddressMode {
  SAMPLER_ADDRESS_REPEAT,
  SAMPLER_ADDRESS_CLAMP_TO_EDGE,
  SAMPLER_ADDRESS_MIRRORED_REPEAT,
  SAMPLER_ADDRESS_CLAMP_TO_BORDER
};

enum SamplerBorderColor {
  SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
  SAMPLER_BORDER_COLOR_OPAQUE_BLACK,
  SAMPLER_BORDER_COLOR_OPAQUE_WHITE
};

struct SamplerDesc {
  SamplerDesc() = default;
  // Set all three filters to the same value.
  SamplerDesc(SamplerFilter filter,
              SamplerAddressMode address_u = SAMPLER_ADDRESS_REPEAT,
              SamplerAddressMode address_v = SAMPLER_ADDRESS_REPEAT);
  SamplerDesc(SamplerFilter min_filter,
              SamplerFilter mag_filter,
              SamplerFilter mip_filter,
              SamplerAddressMode address_u = SAMPLER_ADDRESS_REPEAT,
              SamplerAddressMode address_v = SAMPLER_ADDRESS_REPEAT,
              SamplerAddressMode address_w = SAMPLER_ADDRESS_REPEAT);

  SamplerAddressMode address_u{SAMPLER_ADDRESS_REPEAT};
  SamplerAddressMode address_v{SAMPLER_ADDRESS_REPEAT};
  SamplerAddressMode address_w{SAMPLER_ADDRESS_REPEAT};
  SamplerFilter min_filter{SAMPLER_FILTER_LINEAR};
  SamplerFilter mag_filter{SAMPLER_FILTER_LINEAR};
  SamplerFilter mip_filter{SAMPLER_FILTER_LINEAR};
  // Used by CLAMP_TO_BORDER addressing: (0,0,0,0), (0,0,0,1), or (1,1,1,1).
  SamplerBorderColor border_color{SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK};
  // Comparison samplers are used with RHI_COMPARISON_SAMPLER and SampleCmp*.
  bool compare_enable{false};
  CompareOp compare_op{COMPARE_LESS_EQUAL};
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
