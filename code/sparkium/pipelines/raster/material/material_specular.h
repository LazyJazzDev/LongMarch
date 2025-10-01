#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialSpecular : public Material {
 public:
  MaterialSpecular(sparkium::MaterialSpecular &material);

  graphics::Shader *PixelShader() override;

 private:
  sparkium::MaterialSpecular &material_;
};

}  // namespace sparkium::raster
