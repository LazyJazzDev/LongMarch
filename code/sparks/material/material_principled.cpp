#include "material_principled.h"

#include "sparks/core/core.h"
#include "sparks/core/scene.h"
#include "sparks/material/material_principled.h"

namespace XH {

namespace {
struct RegisteredTextures {
  int normal{-1};
  float y_signal{1.0f};
  int base_color{-1};
  int metallic{-1};
  int specular{-1};
  int roughness{-1};
  int anisotropic{-1};
  int anisotropic_rotation{-1};
};
}  // namespace

MaterialPrincipled::MaterialPrincipled(Core *core, const glm::vec3 &base_color) : Material(core), info{base_color} {
  core_->GraphicsCore()->CreateBuffer(sizeof(Info) + sizeof(RegisteredTextures), graphics::BUFFER_TYPE_STATIC,
                                      &material_buffer_);
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/principled/sampler.hlsl");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/principled/evaluator.hlsli");
}

graphics::Buffer *MaterialPrincipled::Buffer() {
  SyncMaterialData();
  return material_buffer_.get();
}

const CodeLines &MaterialPrincipled::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialPrincipled::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialPrincipled::Update(Scene *scene) {
  RegisteredTextures registered_textures{};
  if (textures.normal) {
    registered_textures.normal = scene->RegisterImage(textures.normal);
    registered_textures.y_signal = textures.normal_reverse_y ? -1.0f : 1.0f;
  }
  if (textures.base_color) {
    registered_textures.base_color = scene->RegisterImage(textures.base_color);
  }
  if (textures.metallic) {
    registered_textures.metallic = scene->RegisterImage(textures.metallic);
  }
  if (textures.specular) {
    registered_textures.specular = scene->RegisterImage(textures.specular);
  }
  if (textures.roughness) {
    registered_textures.roughness = scene->RegisterImage(textures.roughness);
  }
  if (textures.anisotropic) {
    registered_textures.anisotropic = scene->RegisterImage(textures.anisotropic);
  }
  if (textures.anisotropic_rotation) {
    registered_textures.anisotropic_rotation = scene->RegisterImage(textures.anisotropic_rotation);
  }
  material_buffer_->UploadData(&registered_textures, sizeof(RegisteredTextures), sizeof(info));
}

void MaterialPrincipled::SyncMaterialData() {
  material_buffer_->UploadData(&info, sizeof(info));
}

}  // namespace XH
