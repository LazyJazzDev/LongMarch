#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialSpecular : public Material {
 public:
  MaterialSpecular(sparkium::MaterialSpecular &material);

  graphics::Shader *PixelShader() override;

  graphics::Buffer *Buffer() override;

  void Sync() override;

 private:
  sparkium::MaterialSpecular &material_;
  std::unique_ptr<graphics::Shader> pixel_shader_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
};

}  // namespace sparkium::raster
