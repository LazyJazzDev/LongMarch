#pragma once
#include "grassland/graphics/sampler.h"

namespace sparkium::backend::cpu {
using namespace grassland;
using namespace grassland::graphics;

class CpuSampler final : public Sampler {
 public:
  explicit CpuSampler(const SamplerInfo &info);

  SamplerInfo info;
};

}  // namespace sparkium::backend::cpu
