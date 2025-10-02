#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialSpecular : public Material {
 public:
  MaterialSpecular(sparkium::MaterialSpecular &material);

  graphics::Shader *PixelShader() override;

  void Sync() override;

  void BindMaterialResources(graphics::CommandContext *cmd_ctx) override;

 private:
  sparkium::MaterialSpecular &material_;
  std::unique_ptr<graphics::Shader> pixel_shader_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
};

}  // namespace sparkium::raster
