#pragma once
#include "sparkium/pipelines/raytracing/core/material.h"

namespace sparkium::raytracing {

class MaterialPrincipled : public Material {
 public:
  MaterialPrincipled(sparkium::MaterialPrincipled &material);

  using Info = sparkium::MaterialPrincipled::Info;
  using TextureInfo = sparkium::MaterialPrincipled::TextureInfo;

  rhi::IBuffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;
  void Update(Scene *scene) override;

  void SyncMaterialData();

 private:
  sparkium::MaterialPrincipled &material_;
  rhi::ObjPtr<rhi::IBuffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace sparkium::raytracing
