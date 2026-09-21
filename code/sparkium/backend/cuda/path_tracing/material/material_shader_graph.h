#pragma once

#include "sparkium/backend/cuda/path_tracing/core/material.h"

namespace sparkium::cuda_tracing {

class MaterialShaderGraph : public Material {
 public:
  explicit MaterialShaderGraph(sparkium::MaterialShaderGraph &material);
  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;
  const CodeLines *GraphImpl() const override;
  void Update(Scene *scene) override;

 private:
  sparkium::MaterialShaderGraph &material_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace sparkium::cuda_tracing
