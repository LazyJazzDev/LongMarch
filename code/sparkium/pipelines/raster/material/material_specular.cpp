#include "sparkium/pipelines/raster/material/material_specular.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialSpecular::MaterialSpecular(sparkium::MaterialSpecular &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "material/specular/pixel_shader.hlsl", "PSMain", "ps_6_0",
                                      &pixel_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(glm::vec3), graphics::BUFFER_TYPE_STATIC, &material_buffer_);
}

graphics::Shader *MaterialSpecular::PixelShader() {
  return pixel_shader_.get();
}

graphics::Buffer *MaterialSpecular::Buffer() {
  return material_buffer_.get();
}

void MaterialSpecular::Sync() {
  material_buffer_->UploadData(&material_.base_color, sizeof(material_.base_color));
}

}  // namespace sparkium::raster
