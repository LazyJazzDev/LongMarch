#pragma once
#include "sparkium/pipelines/raytracing/core/material.h"

namespace sparkium::raytracing {

class MaterialLight : public Material {
 public:
  MaterialLight(sparkium::MaterialLight &material);

  rhi::IBuffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  void SyncMaterialData();

 private:
  sparkium::MaterialLight &material_;
  rhi::ObjPtr<rhi::IBuffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace sparkium::raytracing
