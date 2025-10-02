#include "sparkium/pipelines/raster/material/material_principled.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialPrincipled::MaterialPrincipled(sparkium::MaterialPrincipled &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "material/light/pixel_shader.hlsl", "PSMain", "ps_6_0",
                                      &pixel_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(material_.info), graphics::BUFFER_TYPE_STATIC, &material_buffer_);
}

graphics::Shader *MaterialPrincipled::PixelShader() {
  return pixel_shader_.get();
}

graphics::Buffer *MaterialPrincipled::Buffer() {
  return material_buffer_.get();
}

void MaterialPrincipled::Sync() {
  material_buffer_->UploadData(&material_.info, sizeof(material_.info));
}

}  // namespace sparkium::raster
