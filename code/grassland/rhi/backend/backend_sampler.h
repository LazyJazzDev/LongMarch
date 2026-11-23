#pragma once
#include "grassland/rhi/rhi_sampler.h"

namespace grassland::rhi::backend {
bool ValidateSamplerDesc(const SamplerDesc &desc);
bool ValidateSamplerFeatures(const SamplerDesc &desc, const PhysicalDeviceMipFeature &features);
}  // namespace grassland::rhi::backend
