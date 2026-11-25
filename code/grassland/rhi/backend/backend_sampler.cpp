#include "grassland/rhi/backend/backend_sampler.h"

#include <cmath>

namespace grassland::rhi::backend {
bool ValidateSamplerDesc(const SamplerDesc &desc) {
  const auto valid_filter = [](SamplerFilter value) {
    return value == SAMPLER_FILTER_NEAREST || value == SAMPLER_FILTER_LINEAR;
  };
  return valid_filter(desc.min_filter) && valid_filter(desc.mag_filter) && valid_filter(desc.mip_filter) &&
         desc.compare_op >= COMPARE_NEVER && desc.compare_op <= COMPARE_ALWAYS && std::isfinite(desc.min_lod) &&
         std::isfinite(desc.max_lod) && std::isfinite(desc.mip_lod_bias) && desc.min_lod >= 0.0f &&
         desc.max_lod >= desc.min_lod && desc.max_anisotropy >= 1 &&
         (desc.max_anisotropy == 1 ||
          (desc.min_filter == SAMPLER_FILTER_LINEAR && desc.mag_filter == SAMPLER_FILTER_LINEAR &&
           desc.mip_filter == SAMPLER_FILTER_LINEAR)) &&
         desc.address_w >= SAMPLER_ADDRESS_REPEAT && desc.address_w <= SAMPLER_ADDRESS_CLAMP_TO_BORDER &&
         desc.address_u >= SAMPLER_ADDRESS_REPEAT && desc.address_u <= SAMPLER_ADDRESS_CLAMP_TO_BORDER &&
         desc.address_v >= SAMPLER_ADDRESS_REPEAT && desc.address_v <= SAMPLER_ADDRESS_CLAMP_TO_BORDER &&
         desc.border_color >= SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK &&
         desc.border_color <= SAMPLER_BORDER_COLOR_OPAQUE_WHITE;
}

bool ValidateSamplerFeatures(const SamplerDesc &desc, const PhysicalDeviceMipFeature &features) {
  return ValidateSamplerDesc(desc) && desc.max_anisotropy <= features.max_anisotropy &&
         desc.mip_lod_bias >= features.min_lod_bias && desc.mip_lod_bias <= features.max_lod_bias;
}
}  // namespace grassland::rhi::backend
