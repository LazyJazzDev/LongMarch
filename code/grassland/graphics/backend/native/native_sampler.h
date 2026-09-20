#pragma once
#include "grassland/graphics/sampler.h"

namespace grassland::graphics::backend {

class NativeSampler final : public Sampler {
 public:
  explicit NativeSampler(const SamplerInfo &info);

  SamplerInfo info;
};

}  // namespace grassland::graphics::backend
