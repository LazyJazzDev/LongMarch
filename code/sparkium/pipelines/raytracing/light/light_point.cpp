#include "sparkium/pipelines/raytracing/light/light_point.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

LightPoint::LightPoint(Core *core,
                       glm::vec3 &position,
                       glm::vec3 &color,
                       float &strength,
                       float &radius,
                       int &soft_falloff,
                       float &sampling_weight)
    : Light(core),
      position(position),
      color(color),
      strength(strength),
      radius(radius),
      soft_falloff(soft_falloff),
      sampling_weight(sampling_weight) {
  core_->GraphicsCore()->CreateBuffer(sizeof(float) * 9, graphics::BUFFER_TYPE_STATIC, &direct_lighting_sampler_data_);
  if (core_->UsesGraphicsRayTracing())
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
  float data[9];
  glm::vec3 power = color * strength;
  float max_power = std::max(std::max(power.r, power.g), power.b);
  std::memcpy(data, &position, sizeof(glm::vec3));
  std::memcpy(data + 3, &power, sizeof(glm::vec3));
  data[6] = sampling_weight >= 0.0f ? sampling_weight : max_power;
  data[7] = std::max(radius, 0.0f);
  std::memcpy(data + 8, &soft_falloff, sizeof(int));
  direct_lighting_sampler_data_->UploadData(data, sizeof(data), 0);
  return sizeof(glm::vec3) + sizeof(glm::vec3);
}

}  // namespace sparkium::raytracing
