#include "sparkium/pipelines/raster/material/material_principled.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

namespace {

struct TextureInfo {
  float y_signal;
  int use_base_color_texture;
  int use_roughness_texture;
  int use_specular_texture;
  int use_metallic_texture;
  int use_normal_texture;
  int use_emission_texture;
};

}  // namespace

MaterialPrincipled::MaterialPrincipled(sparkium::MaterialPrincipled &material)
    : material_(material),
      Material(DedicatedCast(material.GetCore())) {
  pixel_shader_ = core_->GetDevice()->CreateShader("material/principled/pixel_shader.slang", "PSMain",
                                                   rhi::SHADER_STAGE_FRAGMENT, &core_->GetShadersVFS(), {"."}, {});
  material_buffer_ =
      core_->GetDevice()->CreateBuffer({(sizeof(material_.info) + sizeof(TextureInfo) + 255) & ~uint64_t(255)});
  sampler_ = core_->GetDevice()->CreateSampler();
}

rhi::IShader *MaterialPrincipled::PixelShader() {
  return pixel_shader_.Get();
}

void MaterialPrincipled::Sync() {
  TextureInfo info;
  info.y_signal = material_.textures.normal_reverse_y ? -1.0f : 1.0f;
  info.use_base_color_texture = material_.textures.base_color ? 1 : 0;
  info.use_roughness_texture = material_.textures.roughness ? 1 : 0;
  info.use_specular_texture = material_.textures.specular ? 1 : 0;
  info.use_metallic_texture = material_.textures.metallic ? 1 : 0;
  info.use_normal_texture = material_.textures.normal ? 1 : 0;
  info.use_emission_texture = material_.textures.emission ? 1 : 0;
  UploadBuffer(core_->GetDevice(), material_buffer_.Get(), &material_.info, sizeof(material_.info));
  UploadBuffer(core_->GetDevice(), material_buffer_.Get(), &info, sizeof(info), sizeof(material_.info));
}

glm::vec3 MaterialPrincipled::Emission() const {
  return material_.emission_color * material_.emission_strength;
}

void MaterialPrincipled::SetupProgram(rhi::ProgramDesc &program) {
  program.pipeline_layout.groups[0].bindings.push_back({3, rhi::BINDING_TEXTURE2D, 6});  // Material textures
  program.pipeline_layout.groups[0].bindings.push_back({4, rhi::BINDING_SAMPLER, 1});    // Material Data
}

void MaterialPrincipled::BindMaterialResources(rhi::IBindingGroup *group) {
  group->BindResource(2, material_buffer_.Get());
  std::vector<rhi::IImage *> textures(6);
  textures[0] = material_.textures.base_color;
  if (!textures[0])
    textures[0] = core_->GetImage("white");
  textures[1] = material_.textures.roughness;
  if (!textures[1])
    textures[1] = core_->GetImage("white");
  textures[2] = material_.textures.specular;
  if (!textures[2])
    textures[2] = core_->GetImage("black");
  textures[3] = material_.textures.metallic;
  if (!textures[3])
    textures[3] = core_->GetImage("black");
  textures[4] = material_.textures.normal;
  if (!textures[4]) {
    textures[4] = core_->GetImage("normal_default");
  }
  textures[5] = material_.textures.emission;
  if (!textures[5])
    textures[5] = core_->GetImage("white");
  group->BindResource(3, textures);
  group->BindResource(4, sampler_.Get());
}

}  // namespace sparkium::raster
