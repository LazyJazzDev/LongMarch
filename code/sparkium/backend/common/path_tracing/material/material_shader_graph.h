#pragma once

#include "sparkium/backend/common/path_tracing/core/material.h"

namespace sparkium::raytracing {

class MaterialShaderGraph : public Material {
 public:
  explicit MaterialShaderGraph(sparkium::MaterialShaderGraph &material);
  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;
  const CodeLines *GraphImpl() const override;
  void Update(Scene *scene) override;

  graphics::Shader *RenderClosestHitShader() const {
    return closest_hit_shader_.get();
  }

  graphics::Shader *ShadowClosestHitShader() const {
    return shadow_closest_hit_shader_.get();
  }

  graphics::Shader *ShadowAnyHitShader() const {
    return shadow_any_hit_shader_.get();
  }

 private:
  sparkium::MaterialShaderGraph &material_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
  std::unique_ptr<graphics::Shader> closest_hit_shader_;
  std::unique_ptr<graphics::Shader> shadow_closest_hit_shader_;
  std::unique_ptr<graphics::Shader> shadow_any_hit_shader_;
};

}  // namespace sparkium::raytracing
