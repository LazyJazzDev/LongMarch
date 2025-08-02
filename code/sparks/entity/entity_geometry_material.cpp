#include "sparks/entity/entity_geometry_material.h"

#include "sparks/core/core.h"
#include "sparks/core/geometry.h"
#include "sparks/core/material.h"
#include "sparks/core/scene.h"

namespace sparks {

EntityGeometryMaterial::EntityGeometryMaterial(Core *core,
                                               Geometry *geometry,
                                               Material *material,
                                               const glm::mat4x3 &transformation)
    : Entity(core) {
  geometry_ = geometry;
  material_ = material;
  transformation_ = transformation;

  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("material_sampler.hlsli", material->SamplerImpl());
  vfs.WriteFile("entity_chit.hlsl", geometry_->ClosestHitShaderImpl());
  core_->GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_3", {"-I."},
                                      &closest_hit_shader_);
  core_->GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_3", {"-I."},
                                      &shadow_closest_hit_shader_);
}

void EntityGeometryMaterial::Update(Scene *scene) {
  scene->RegisterInstance(geometry_->BLAS(), transformation_,
                          scene->RegisterHitGroup({{closest_hit_shader_.get()}, {shadow_closest_hit_shader_.get()}}),
                          scene->RegisterBuffer(geometry_->Buffer()), scene->RegisterBuffer(material_->Buffer()));
}

}  // namespace sparks
