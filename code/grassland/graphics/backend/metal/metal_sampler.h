#pragma once
#include "grassland/graphics/backend/metal/metal_util.h"

namespace grassland::graphics::backend {

class MetalSampler : public Sampler {
 public:
  MetalSampler(MetalCore *core, const SamplerInfo &info);
  NS::SharedPtr<MTL::SamplerState> state;
};

}  // namespace grassland::graphics::backend
