#pragma once
#include "sparks/core/light.h"

namespace sparks {

class LightGeometrySurface : public Light {
 public:
  struct BlellochScanMetadata {
    uint32_t offset;
    uint32_t stride;
    uint32_t element_count;
    uint32_t padding[61];
  };

  LightGeometrySurface(Core *core, Geometry *geometry, Surface *surface, const glm::mat4x3 &transform);
  graphics::Shader *SamplerShader() override;
  graphics::Buffer *SamplerData() override;
  void Update(Scene *scene, uint32_t light_index) override;

 private:
  Geometry *geometry_;
  Surface *surface_;
  glm::mat4x3 transform_;
  std::unique_ptr<graphics::Shader> direct_lighting_sampler_;
  std::unique_ptr<graphics::Buffer> direct_lighting_sampler_data_;

  graphics::ComputeProgram *blelloch_scan_up_program;
  graphics::ComputeProgram *blelloch_scan_down_program;

  std::vector<BlellochScanMetadata> metadatas;
  std::unique_ptr<graphics::Buffer> metadata_buffer;

  std::unique_ptr<graphics::Shader> gather_primitive_power_shader;
  std::unique_ptr<graphics::ComputeProgram> gather_primitive_power_program;

  // float3x4 transform
  // uint num_primitive
  // uint cdf[num_primitive];
};

}  // namespace sparks
