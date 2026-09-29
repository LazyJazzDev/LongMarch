#pragma once
#include "grassland/graphics/backend/webgpu/webgpu_util.h"

namespace grassland::graphics::backend {

class WebGPUSampler : public Sampler {
 public:
  WebGPUSampler(WebGPUCore *core, const SamplerInfo &info);

  const wgpu::Sampler &Handle() const {
    return sampler_;
  }

 private:
  wgpu::Sampler sampler_;
};

}  // namespace grassland::graphics::backend
