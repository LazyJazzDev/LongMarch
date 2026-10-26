#pragma once
#include "sparkium/pipelines/raytracing/core/light.h"

namespace sparkium::raytracing {

class LightGeometryMaterial : public Light {
 public:
  LightGeometryMaterial(Core *core, Geometry *geometry, Material *material, const glm::mat4x3 &transform);
  int SamplerShader(Scene *scene) override;
  rhi::IBuffer *SamplerData() override;
  uint32_t SamplerPreprocess(rhi::ICommandContext *cmd_ctx) override;

  const glm::mat4x3 &transform;

 private:
  Geometry *geometry_;
  Material *material_;
  rhi::ObjPtr<rhi::IShader> direct_lighting_sampler_;
  rhi::ObjPtr<rhi::IBuffer> direct_lighting_sampler_data_;

  rhi::IComputeProgram *blelloch_scan_up_program_;
  rhi::IComputeProgram *blelloch_scan_down_program_;

  std::vector<BlellochScanMetadata> metadatas_;
  rhi::ObjPtr<rhi::IBuffer> metadata_buffer_;

  rhi::IComputeProgram *gather_primitive_power_program_;

  // float3x4 transform
  // uint num_primitive
  // uint cdf[num_primitive];
  rhi::ObjPtr<rhi::IBindingGroup> gather_group_, scan_group_;
};

}  // namespace sparkium::raytracing
