#pragma once
#include "sparkium/pipelines/raytracing/core/light.h"

namespace sparkium::raytracing {

class LightGeometryMaterial : public Light {
 public:
  LightGeometryMaterial(Core *core, Geometry *geometry, Material *material, const glm::mat4x3 &transform);
  int SamplerShader(Scene *scene) override;
  graphics::Buffer *SamplerData() override;
  uint32_t SamplerPreprocess(graphics::CommandContext *cmd_ctx) override;

  const glm::mat4x3 &transform;

 private:
  Geometry *geometry_;
  Material *material_;
  std::unique_ptr<graphics::Shader> direct_lighting_sampler_;
  std::unique_ptr<graphics::Buffer> direct_lighting_sampler_data_;

  graphics::ComputeProgram *blelloch_scan_up_program_;
  graphics::ComputeProgram *blelloch_scan_down_program_;

  std::vector<BlellochScanMetadata> metadatas_;
  std::unique_ptr<graphics::Buffer> metadata_buffer_;

  std::unique_ptr<graphics::Shader> gather_primitive_power_shader_;
  std::unique_ptr<graphics::ComputeProgram> gather_primitive_power_program_;

  // float3x4 transform
  // uint num_primitive
  // uint cdf[num_primitive];
};

}  // namespace sparkium::raytracing
