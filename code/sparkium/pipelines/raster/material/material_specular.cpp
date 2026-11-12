#include "sparkium/pipelines/raster/material/material_specular.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialSpecular::MaterialSpecular(sparkium::MaterialSpecular &material)
    : material_(material),
      Material(DedicatedCast(material.GetCore())) {
  auto shaders_vfs = core_->GetShadersVFS();
  pixel_shader_ = core_->GetDevice()->CreateShader("material/specular/pixel_shader.slang", "PSMain",
                                                   rhi::SHADER_STAGE_FRAGMENT, &shaders_vfs, {"."}, {});
  material_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(glm::vec3) + 255) & ~uint64_t(255)});
}

rhi::IShader *MaterialSpecular::PixelShader() {
  return pixel_shader_.Get();
}

void MaterialSpecular::Sync() {
  core_->QueueUpload(material_buffer_.Get(), &material_.base_color, sizeof(material_.base_color));
}

void MaterialSpecular::BindMaterialResources(rhi::IBindingGroup *group) {
  group->BindResource(2, material_buffer_.Get());
}

}  // namespace sparkium::raster
