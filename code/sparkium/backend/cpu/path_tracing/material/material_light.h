#pragma once
#include "sparkium/backend/cpu/path_tracing/core/material.h"

namespace sparkium::cpu_tracing {

class MaterialLight : public Material {
 public:
  MaterialLight(sparkium::MaterialLight &material);

  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  void SyncMaterialData();

 private:
  sparkium::MaterialLight &material_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace sparkium::cpu_tracing
