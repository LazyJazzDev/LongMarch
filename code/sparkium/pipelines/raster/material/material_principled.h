#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialPrincipled : public Material {
 public:
  MaterialPrincipled(sparkium::MaterialPrincipled &material);

  rhi::IShader *PixelShader() override;

  void Sync() override;

  glm::vec3 Emission() const override;

  void SetupProgram(rhi::ProgramDesc &program) override;

  void BindMaterialResources(rhi::IBindingGroup *group) override;

 private:
  sparkium::MaterialPrincipled &material_;
  rhi::ObjPtr<rhi::IShader> pixel_shader_;
  rhi::ObjPtr<rhi::IBuffer> material_buffer_;
  rhi::ObjPtr<rhi::ISampler> sampler_;
};

}  // namespace sparkium::raster
