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
};

}  // namespace

MaterialPrincipled::MaterialPrincipled(sparkium::MaterialPrincipled &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "material/principled/pixel_shader.hlsl", "PSMain",
                                      "ps_6_0", {"-I."}, &pixel_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(material_.info) + sizeof(TextureInfo), graphics::BUFFER_TYPE_STATIC,
                                      &material_buffer_);
  core_->GraphicsCore()->CreateSampler(graphics::SamplerInfo{}, &sampler_);
}

graphics::Shader *MaterialPrincipled::PixelShader() {
  return pixel_shader_.get();
}

void MaterialPrincipled::Sync() {
  TextureInfo info;
  info.y_signal = material_.textures.normal_reverse_y ? -1.0f : 1.0f;
  info.use_base_color_texture = material_.textures.base_color ? 1 : 0;
  info.use_roughness_texture = material_.textures.roughness ? 1 : 0;
  info.use_specular_texture = material_.textures.specular ? 1 : 0;
  info.use_metallic_texture = material_.textures.metallic ? 1 : 0;
  info.use_normal_texture = material_.textures.normal ? 1 : 0;
  material_buffer_->UploadData(&material_.info, sizeof(material_.info));
  material_buffer_->UploadData(&info, sizeof(info), sizeof(material_.info));
}

glm::vec3 MaterialPrincipled::Emission() const {
  return material_.emission_color * material_.emission_strength;
}

void MaterialPrincipled::SetupProgram(graphics::Program *program) {
  program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 5);    // Material Data
  program->AddResourceBinding(graphics::RESOURCE_TYPE_SAMPLER, 1);  // Material Data
}

void MaterialPrincipled::BindMaterialResources(graphics::CommandContext *cmd_ctx) {
  cmd_ctx->CmdBindResources(2, {material_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
  std::vector<graphics::Image *> textures(5);
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
  cmd_ctx->CmdBindResources(3, textures, graphics::BIND_POINT_GRAPHICS);
  cmd_ctx->CmdBindResources(4, {sampler_.get()}, graphics::BIND_POINT_GRAPHICS);
}

}  // namespace sparkium::raster
