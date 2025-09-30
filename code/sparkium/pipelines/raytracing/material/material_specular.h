#pragma once
#include "sparkium/pipelines/raytracing/core/material.h"

namespace sparkium::raytracing {

class MaterialSpecular : public Material {
 public:
  MaterialSpecular(sparkium::MaterialSpecular &material);

  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  void SyncMaterialData();

 private:
  sparkium::MaterialSpecular &material_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace sparkium::raytracing
