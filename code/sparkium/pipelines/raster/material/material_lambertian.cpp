#include "sparkium/pipelines/raster/material/material_lambertian.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialLambertian::MaterialLambertian(sparkium::MaterialLambertian &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "material/lambertian/pixel_shader.hlsl", "PSMain",
                                      "ps_6_0", &pixel_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(glm::vec3) + sizeof(glm::vec3), graphics::BUFFER_TYPE_STATIC,
                                      &material_buffer_);
}

graphics::Shader *MaterialLambertian::PixelShader() {
  return pixel_shader_.get();
}

void MaterialLambertian::Sync() {
  float data[6];
  data[0] = material_.base_color.r;
  data[1] = material_.base_color.g;
  data[2] = material_.base_color.b;
  data[3] = material_.emission.r;
  data[4] = material_.emission.g;
  data[5] = material_.emission.b;
  material_buffer_->UploadData(data, sizeof(data));
}

glm::vec3 MaterialLambertian::Emission() const {
  return material_.emission;
}

void MaterialLambertian::BindMaterialResources(graphics::CommandContext *cmd_ctx) {
  cmd_ctx->CmdBindResources(2, {material_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
}

}  // namespace sparkium::raster
