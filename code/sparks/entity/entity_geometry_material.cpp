#include "sparks/entity/entity_geometry_material.h"

#include "sparks/core/core.h"
#include "sparks/core/geometry.h"
#include "sparks/core/material.h"
#include "sparks/core/scene.h"
#include "sparks/geometry/geometries.h"
#include "sparks/material/materials.h"

namespace sparks {

EntityGeometryMaterial::EntityGeometryMaterial(Core *core,
                                               Geometry *geometry,
                                               Material *material,
                                               const glm::mat4x3 &transformation)
    : Entity(core), light_geom_mat_(core, geometry, material, transformation) {
  geometry_ = geometry;
  material_ = material;
  transformation_ = transformation;

  if (dynamic_cast<GeometryMesh *>(geometry_)) {
    if (dynamic_cast<MaterialLambertian *>(material_)) {
      hit_groups_.render_group.closest_hit_shader = core_->GetShader("mesh_lambertian_chit");
      hit_groups_.shadow_group.closest_hit_shader = core_->GetShader("mesh_lambertian_shadow_chit");
    } else if (dynamic_cast<MaterialLight *>(material_)) {
      hit_groups_.render_group.closest_hit_shader = core_->GetShader("mesh_light_chit");
      hit_groups_.shadow_group.closest_hit_shader = core_->GetShader("mesh_light_shadow_chit");
    } else if (dynamic_cast<MaterialPrincipled *>(material_)) {
      hit_groups_.render_group.closest_hit_shader = core_->GetShader("mesh_principled_chit");
      hit_groups_.shadow_group.closest_hit_shader = core_->GetShader("mesh_principled_shadow_chit");
    } else if (dynamic_cast<MaterialSpecular *>(material_)) {
      hit_groups_.render_group.closest_hit_shader = core_->GetShader("mesh_specular_chit");
      hit_groups_.shadow_group.closest_hit_shader = core_->GetShader("mesh_specular_shadow_chit");
    }
  }
  if (!hit_groups_.render_group.closest_hit_shader) {
    auto vfs = core_->GetShadersVFS();
    vfs.WriteFile("material_sampler.hlsli", material_->SamplerImpl());
    vfs.WriteFile("entity_chit.hlsl", geometry_->ClosestHitShaderImpl());
    core_->GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."},
                                        &closest_hit_shader_);
    core_->GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."},
                                        &shadow_closest_hit_shader_);
    hit_groups_.render_group.closest_hit_shader = closest_hit_shader_.get();
    hit_groups_.shadow_group.closest_hit_shader = shadow_closest_hit_shader_.get();
  }
}

void EntityGeometryMaterial::Update(Scene *scene) {
  material_->Update(scene);
  int32_t light_index = scene->RegisterLight(&light_geom_mat_);
  int32_t instance_index = scene->RegisterInstance(
      geometry_->BLAS(), transformation_, scene->RegisterHitGroup(hit_groups_),
      scene->RegisterBuffer(geometry_->Buffer()), scene->RegisterBuffer(material_->Buffer()), light_index);
  scene->LightCustomIndex(light_index) = instance_index;
}

void EntityGeometryMaterial::SetTransformation(const glm::mat4x3 &transformation) {
  transformation_ = transformation;
  light_geom_mat_.transform = transformation_;
}

}  // namespace sparks
