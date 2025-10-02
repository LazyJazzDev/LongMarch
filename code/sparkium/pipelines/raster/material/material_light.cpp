#include "sparkium/pipelines/raster/material/material_light.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialLight::MaterialLight(sparkium::MaterialLight &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "material/light/pixel_shader.hlsl", "PSMain", "ps_6_0",
                                      &pixel_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(glm::vec3), graphics::BUFFER_TYPE_STATIC, &material_buffer_);
}

graphics::Shader *MaterialLight::PixelShader() {
  return pixel_shader_.get();
}

void MaterialLight::Sync() {
  material_buffer_->UploadData(&material_.emission, sizeof(material_.emission));
}

glm::vec3 MaterialLight::Emission() const {
  return material_.emission;
}

void MaterialLight::BindMaterialResources(graphics::CommandContext *cmd_ctx) {
  cmd_ctx->CmdBindResources(2, {material_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
}

}  // namespace sparkium::raster
