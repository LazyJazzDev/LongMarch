#include "sparkium/pipelines/raytracing/material/material_principled.h"

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

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

MaterialPrincipled::MaterialPrincipled(sparkium::MaterialPrincipled &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
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
  if (material_.textures.normal) {
    registered_textures.normal = scene->RegisterImage(material_.textures.normal);
    registered_textures.y_signal = material_.textures.normal_reverse_y ? -1.0f : 1.0f;
  }
  if (material_.textures.base_color) {
    registered_textures.base_color = scene->RegisterImage(material_.textures.base_color);
  }
  if (material_.textures.metallic) {
    registered_textures.metallic = scene->RegisterImage(material_.textures.metallic);
  }
  if (material_.textures.specular) {
    registered_textures.specular = scene->RegisterImage(material_.textures.specular);
  }
  if (material_.textures.roughness) {
    registered_textures.roughness = scene->RegisterImage(material_.textures.roughness);
  }
  if (material_.textures.anisotropic) {
    registered_textures.anisotropic = scene->RegisterImage(material_.textures.anisotropic);
  }
  if (material_.textures.anisotropic_rotation) {
    registered_textures.anisotropic_rotation = scene->RegisterImage(material_.textures.anisotropic_rotation);
  }
  material_buffer_->UploadData(&registered_textures, sizeof(RegisteredTextures), sizeof(material_.info));
}

void MaterialPrincipled::SyncMaterialData() {
  material_buffer_->UploadData(&material_.info, sizeof(material_.info));
}

}  // namespace sparkium::raytracing
