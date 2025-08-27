#pragma once
#include "sparks/core/light.h"

namespace sparks {

class LightPoint : public Light {
 public:
  LightPoint(Core *core, const glm::vec3 &position, const glm::vec3 &color, float strength);

  graphics::Buffer *SamplerData() override;
  uint32_t SamplerPreprocess(graphics::CommandContext *cmd_ctx) override;
  const CodeLines &SamplerImpl() const override;

  glm::vec3 position;
  glm::vec3 color;
  float strength;

 private:
  std::unique_ptr<graphics::Buffer> direct_lighting_sampler_data_;
  CodeLines sampler_implementation_;
};

}  // namespace sparks
