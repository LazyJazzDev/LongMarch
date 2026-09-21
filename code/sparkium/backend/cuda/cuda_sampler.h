#pragma once
#include "grassland/graphics/sampler.h"

namespace sparkium::backend::cuda {
using namespace grassland;
using namespace grassland::graphics;

class CudaSampler final : public Sampler {
 public:
  explicit CudaSampler(const SamplerInfo &info);

  SamplerInfo info;
};

}  // namespace sparkium::backend::cuda
