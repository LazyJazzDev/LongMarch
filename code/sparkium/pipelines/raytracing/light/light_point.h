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
  rhi::IBuffer *SamplerData() override;
  uint32_t SamplerPreprocess(rhi::ICommandContext *cmd_ctx) override;

  glm::vec3 &position;
  glm::vec3 &color;
  float &strength;
  float &radius;
  int &soft_falloff;
  float &sampling_weight;

 private:
  rhi::ObjPtr<rhi::IBuffer> direct_lighting_sampler_data_;
};

}  // namespace sparkium::raytracing
