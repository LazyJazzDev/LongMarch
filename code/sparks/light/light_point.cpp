#include "sparks/light/light_point.h"

#include "sparks/core/core.h"

namespace sparks {

LightPoint::LightPoint(Core *core, const glm::vec3 &position, const glm::vec3 &color, float strength)
    : Light(core), position(position), color(color), strength(strength) {
  core_->GraphicsCore()->CreateBuffer(sizeof(glm::vec3) + sizeof(glm::vec3) + sizeof(float),
                                      graphics::BUFFER_TYPE_STATIC, &direct_lighting_sampler_data_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "light/point/direct_lighting_sampler.hlsl",
                                      "SampleDirectLightingCallable", "lib_6_5", {"-I."}, &direct_lighting_sampler_);
}

int LightPoint::SamplerShader(Scene *scene) {
  return 0x1000000;
}

graphics::Buffer *LightPoint::SamplerData() {
  return direct_lighting_sampler_data_.get();
}

uint32_t LightPoint::SamplerPreprocess(graphics::CommandContext *cmd_ctx) {
  float data[7];
  glm::vec3 power = color * strength;
  float max_power = std::max(std::max(power.r, power.g), power.b);
  std::memcpy(data, &position, sizeof(glm::vec3));
  std::memcpy(data + 3, &power, sizeof(glm::vec3));
  data[6] = max_power;
  direct_lighting_sampler_data_->UploadData(data, sizeof(data), 0);
  return sizeof(glm::vec3) + sizeof(glm::vec3);
}

}  // namespace sparks
