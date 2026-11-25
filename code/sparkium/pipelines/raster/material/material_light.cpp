#include "sparkium/pipelines/raster/material/material_light.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialLight::MaterialLight(sparkium::MaterialLight &material)
    : material_(material),
      Material(DedicatedCast(material.GetCore())) {
  auto shaders_vfs = core_->GetShadersVFS();
  pixel_shader_ = core_->GetDevice()->CreateShader("material/light/pixel_shader.slang", "PSMain",
                                                   rhi::SHADER_STAGE_FRAGMENT, &shaders_vfs, {"."}, {});
  material_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(glm::vec3) + 255) & ~uint64_t(255)});
  if (!(pixel_shader_ && material_buffer_))
    core_->ReportFailure("Cannot create raster material resources");
}

rhi::IShader *MaterialLight::PixelShader() {
  return pixel_shader_.Get();
}

void MaterialLight::Sync() {
  core_->QueueUpload(material_buffer_.Get(), &material_.emission, sizeof(material_.emission));
}

glm::vec3 MaterialLight::Emission() const {
  return material_.emission;
}

bool MaterialLight::BindMaterialResources(rhi::IBindingGroup *group) {
  return group && group->BindResource(2, material_buffer_.Get());
}

}  // namespace sparkium::raster
