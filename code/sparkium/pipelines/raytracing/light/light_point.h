#pragma once
#include "sparkium/pipelines/raytracing/core/light.h"

namespace sparkium::raytracing {

class LightPoint : public Light {
 public:
  LightPoint(Core *core,
             glm::vec3 &position,
             glm::vec3 &color,
             float &strength,
             float &radius,
             int &soft_falloff,
             float &sampling_weight);

  int SamplerShader(Scene *scene) override;
  graphics::Buffer *SamplerData() override;
  uint32_t SamplerPreprocess(graphics::CommandContext *cmd_ctx) override;

  glm::vec3 &position;
  glm::vec3 &color;
  float &strength;
  float &radius;
  int &soft_falloff;
  float &sampling_weight;

 private:
  std::unique_ptr<graphics::Shader> direct_lighting_sampler_;
  std::unique_ptr<graphics::Buffer> direct_lighting_sampler_data_;
};

}  // namespace sparkium::raytracing
