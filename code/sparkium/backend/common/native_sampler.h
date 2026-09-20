#pragma once
#include "grassland/graphics/sampler.h"

namespace sparkium::backend {
using namespace grassland;
using namespace grassland::graphics;

class NativeSampler final : public Sampler {
 public:
  explicit NativeSampler(const SamplerInfo &info);

  SamplerInfo info;
};

}  // namespace sparkium::backend
