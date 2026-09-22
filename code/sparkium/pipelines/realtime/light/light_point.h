#pragma once
#include "sparkium/pipelines/realtime/core/light.h"

namespace sparkium::realtime {

class LightPoint : public Light {
 public:
  LightPoint(Core *core,
             glm::vec3 &position,
             glm::vec3 &color,
             float &strength,
             float &radius,
             int &soft_falloff,
             float &sampling_weight);

  int SamplerKind() override;
  graphics::Buffer *SamplerData() override;
  uint32_t SamplerPreprocess(graphics::CommandContext *cmd_ctx) override;

  glm::vec3 &position;
  glm::vec3 &color;
  float &strength;
  float &radius;
  int &soft_falloff;
  float &sampling_weight;

 private:
  std::unique_ptr<graphics::Buffer> direct_lighting_sampler_data_;
};

}  // namespace sparkium::realtime
