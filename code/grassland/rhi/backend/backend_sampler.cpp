#include "grassland/rhi/backend/backend_sampler.h"

#include <cmath>

namespace grassland::rhi::backend {
bool ValidateSamplerDesc(const SamplerDesc &desc) {
  return (!desc.comparison || (*desc.comparison >= COMPARE_NEVER && *desc.comparison <= COMPARE_ALWAYS)) &&
         desc.filter >= SAMPLER_FILTER_NEAREST && desc.filter <= SAMPLER_FILTER_LINEAR &&
         desc.address_u >= SAMPLER_ADDRESS_REPEAT && desc.address_u <= SAMPLER_ADDRESS_MIRRORED_REPEAT &&
         desc.address_v >= SAMPLER_ADDRESS_REPEAT && desc.address_v <= SAMPLER_ADDRESS_MIRRORED_REPEAT &&
         desc.mip_filter >= SAMPLER_FILTER_NEAREST && desc.mip_filter <= SAMPLER_FILTER_LINEAR &&
         std::isfinite(desc.min_lod) && std::isfinite(desc.max_lod) && desc.min_lod >= 0 &&
         desc.max_lod >= desc.min_lod && desc.max_anisotropy >= 1 && desc.max_anisotropy <= 16 &&
         (desc.max_anisotropy == 1 ||
          (desc.filter == SAMPLER_FILTER_LINEAR && desc.mip_filter == SAMPLER_FILTER_LINEAR));
}
}  // namespace grassland::rhi::backend
