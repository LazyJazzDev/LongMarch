#pragma once
#include "sparkium/core/light.h"

namespace sparkium {

class LightPoint : public Light {
 public:
  LightPoint(Core *core, const glm::vec3 &position, const glm::vec3 &color, float strength);

  int SamplerShader(Scene *scene) override;
  graphics::Buffer *SamplerData() override;
  uint32_t SamplerPreprocess(graphics::CommandContext *cmd_ctx) override;

  glm::vec3 position;
  glm::vec3 color;
  float strength;

 private:
  std::unique_ptr<graphics::Shader> direct_lighting_sampler_;
  std::unique_ptr<graphics::Buffer> direct_lighting_sampler_data_;
};

}  // namespace sparkium
