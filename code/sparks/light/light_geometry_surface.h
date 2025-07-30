#pragma once
#include "sparks/core/light.h"

namespace sparks {

class LightGeometrySurface : public Light {
 public:
  LightGeometrySurface(Core *core);
  graphics::Shader *SamplerShader() override;
  graphics::Buffer *SamplerData() override;

 private:
  std::unique_ptr<graphics::Shader> direct_lighting_sampler_;
  std::unique_ptr<graphics::Buffer> direct_lighting_sampler_data_;
};

}  // namespace sparks
