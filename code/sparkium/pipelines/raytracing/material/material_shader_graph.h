#pragma once

#include "sparkium/pipelines/raytracing/core/material.h"

namespace sparkium::raytracing {

class MaterialShaderGraph : public Material {
 public:
  explicit MaterialShaderGraph(sparkium::MaterialShaderGraph &material);
  void PrepareHitShaders();
  rhi::IBuffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;
  const CodeLines *GraphImpl() const override;
  void Update(Scene *scene) override;

  rhi::IShader *RenderClosestHitShader() const {
    return closest_hit_shader_.Get();
  }

  rhi::IShader *ShadowClosestHitShader() const {
    return shadow_closest_hit_shader_.Get();
  }

  rhi::IShader *ShadowAnyHitShader() const {
    return shadow_any_hit_shader_.Get();
  }

 private:
  sparkium::MaterialShaderGraph &material_;
  rhi::ObjPtr<rhi::IBuffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
  rhi::ObjPtr<rhi::IShader> closest_hit_shader_;
  rhi::ObjPtr<rhi::IShader> shadow_closest_hit_shader_;
  rhi::ObjPtr<rhi::IShader> shadow_any_hit_shader_;
};

}  // namespace sparkium::raytracing
