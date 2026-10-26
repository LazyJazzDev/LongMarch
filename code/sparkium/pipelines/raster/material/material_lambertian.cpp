#include "sparkium/pipelines/raster/material/material_lambertian.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialLambertian::MaterialLambertian(sparkium::MaterialLambertian &material)
    : material_(material),
      Material(DedicatedCast(material.GetCore())) {
  pixel_shader_ = core_->GetDevice()->CreateShader("material/lambertian/pixel_shader.slang", "PSMain",
                                                   rhi::SHADER_STAGE_FRAGMENT, &core_->GetShadersVFS(), {"."}, {});
  material_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(glm::vec3) + sizeof(glm::vec3) + 255) & ~uint64_t(255)});
}

rhi::IShader *MaterialLambertian::PixelShader() {
  return pixel_shader_.Get();
}

void MaterialLambertian::Sync() {
  float data[6];
  data[0] = material_.base_color.r;
  data[1] = material_.base_color.g;
  data[2] = material_.base_color.b;
  data[3] = material_.emission.r;
  data[4] = material_.emission.g;
  data[5] = material_.emission.b;
  UploadBuffer(core_->GetDevice(), material_buffer_.Get(), data, sizeof(data));
}

glm::vec3 MaterialLambertian::Emission() const {
  return material_.emission;
}

void MaterialLambertian::BindMaterialResources(rhi::IBindingGroup *group) {
  group->BindResource(2, material_buffer_.Get());
}

}  // namespace sparkium::raster
