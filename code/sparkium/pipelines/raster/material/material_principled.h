#pragma once
#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

class MaterialPrincipled : public Material {
 public:
  MaterialPrincipled(sparkium::MaterialPrincipled &material);

  graphics::Shader *PixelShader() override;

  void Sync() override;

  glm::vec3 Emission() const override;

  void SetupProgram(graphics::Program *program) override;

  void BindMaterialResources(graphics::CommandContext *cmd_ctx) override;

 private:
  sparkium::MaterialPrincipled &material_;
  std::unique_ptr<graphics::Shader> pixel_shader_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
  std::unique_ptr<graphics::Sampler> sampler_;
};

}  // namespace sparkium::raster
