#include "grassland/rhi/backend/backend_sampler.h"

namespace grassland::rhi::backend {
bool IBackendSampler::Validate(const SamplerDesc &desc) {
  return desc.filter >= SAMPLER_FILTER_NEAREST && desc.filter <= SAMPLER_FILTER_LINEAR &&
         desc.address_u >= SAMPLER_ADDRESS_REPEAT && desc.address_u <= SAMPLER_ADDRESS_MIRRORED_REPEAT &&
         desc.address_v >= SAMPLER_ADDRESS_REPEAT && desc.address_v <= SAMPLER_ADDRESS_MIRRORED_REPEAT;
}
}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendSampler, "34cc865a-50d9-49f8-b8db-591bc452621f")
