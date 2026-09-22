#pragma once
#include "sparkium/pipelines/realtime/core/material.h"

namespace sparkium::realtime {

class MaterialLambertian : public Material {
 public:
  MaterialLambertian(sparkium::MaterialLambertian &material);

  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  void SyncMaterialData();

 private:
  sparkium::MaterialLambertian &material_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace sparkium::realtime
