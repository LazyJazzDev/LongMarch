#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialLight : public Material {
 public:
  MaterialLight(sparkium::MaterialLight &material);

  graphics::Shader *PixelShader() override;

 private:
  sparkium::MaterialLight &material_;
};

}  // namespace sparkium::raster
