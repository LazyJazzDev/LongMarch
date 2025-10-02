#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialLambertian : public Material {
 public:
  MaterialLambertian(sparkium::MaterialLambertian &material);

  graphics::Shader *PixelShader() override;
  void Sync() override;
  glm::vec3 Emission() const override;
  void BindMaterialResources(graphics::CommandContext *cmd_ctx) override;

 private:
  sparkium::MaterialLambertian &material_;
  std::unique_ptr<graphics::Shader> pixel_shader_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
};

}  // namespace sparkium::raster
