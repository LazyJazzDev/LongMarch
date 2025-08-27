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
      transform(transform) {
}

void EntityGeometryLight::Update(Scene *scene) {
  material_light_.emission = emission;
  material_light_.two_sided = two_sided;
  material_light_.block_ray = block_ray;
  light_geom_mat_.transform = transform;
  auto geom_reg = scene->RegisterGeometry(geometry_);
  auto mat_reg = scene->RegisterMaterial(&material_light_);
  int32_t light_index = scene->RegisterLight(&light_geom_mat_);
  int32_t instance_index = scene->RegisterInstance(geom_reg, transform, mat_reg, light_index);
  scene->LightCustomIndex(light_index) = instance_index;
}

}  // namespace sparks
