#pragma once
#include "grassland/rhi/rhi_sampler.h"

namespace grassland::rhi::backend {
bool ValidateSamplerFeatures(const SamplerDesc &desc, const PhysicalDeviceMipFeature &features);
}  // namespace grassland::rhi::backend
