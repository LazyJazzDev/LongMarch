#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialSpecular : public Material {
 public:
  MaterialSpecular(sparkium::MaterialSpecular &material);

  rhi::IShader *PixelShader() override;

  void Sync() override;

  void BindMaterialResources(rhi::IBindingGroup *group) override;

 private:
  sparkium::MaterialSpecular &material_;
  rhi::ObjPtr<rhi::IShader> pixel_shader_;
  rhi::ObjPtr<rhi::IBuffer> material_buffer_;
};

}  // namespace sparkium::raster
