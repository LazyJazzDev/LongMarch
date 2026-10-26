#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialLambertian : public Material {
 public:
  MaterialLambertian(sparkium::MaterialLambertian &material);

  rhi::IShader *PixelShader() override;
  void Sync() override;
  glm::vec3 Emission() const override;
  void BindMaterialResources(rhi::IBindingGroup *group) override;

 private:
  sparkium::MaterialLambertian &material_;
  rhi::ObjPtr<rhi::IShader> pixel_shader_;
  rhi::ObjPtr<rhi::IBuffer> material_buffer_;
};

}  // namespace sparkium::raster
