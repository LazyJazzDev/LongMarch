#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialPrincipled : public Material {
 public:
  MaterialPrincipled(sparkium::MaterialPrincipled &material);

  graphics::Shader *PixelShader() override;

 private:
  sparkium::MaterialPrincipled &material_;
};

}  // namespace sparkium::raster
