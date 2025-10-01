#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialLambertian : public Material {
 public:
  MaterialLambertian(sparkium::MaterialLambertian &material);

  graphics::Shader *PixelShader() override;

 private:
  sparkium::MaterialLambertian &material_;
};

}  // namespace sparkium::raster
