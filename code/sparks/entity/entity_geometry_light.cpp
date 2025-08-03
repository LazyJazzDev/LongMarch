#include "sparks/entity/entity_geometry_light.h"

#include "sparks/core/core.h"
#include "sparks/core/geometry.h"
#include "sparks/core/material.h"
#include "sparks/core/scene.h"

namespace sparks {

EntityGeometryLight::EntityGeometryLight(Core *core,
                                         Geometry *geometry,
                                         const glm::vec3 &emission,
                                         bool two_sided,
                                         bool block_ray,
                                         const glm::mat4x3 &transform)
    : Entity(core),
      material_light_(core, emission, two_sided, block_ray),
      light_geom_mat_(core, geometry, &material_light_, transform),
      geometry_(geometry),
      emission(emission),
      two_sided(two_sided),
      block_ray(block_ray),
      transformation_(transform) {
  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("entity_chit.hlsl", geometry_->ClosestHitShaderImpl());
  vfs.WriteFile("material_sampler.hlsli", material_light_.SamplerImpl());
  core_->GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_3", {"-I."},
                                      &closest_hit_shader_);
  core_->GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_3", {"-I."},
                                      &shadow_closest_hit_shader_);
}

void EntityGeometryLight::Update(Scene *scene) {
  material_light_.emission = emission;
  material_light_.two_sided = two_sided;
  material_light_.block_ray = block_ray;
  int32_t light_index = scene->RegisterLight(&light_geom_mat_);
  int32_t instance_index = scene->RegisterInstance(
      geometry_->BLAS(), transformation_,
      scene->RegisterHitGroup({{closest_hit_shader_.get()}, {shadow_closest_hit_shader_.get()}}),
      scene->RegisterBuffer(geometry_->Buffer()), scene->RegisterBuffer(material_light_.Buffer()), light_index);
  scene->LightCustomIndex(light_index) = instance_index;
}

}  // namespace sparks
