#include "sparks/entity/entity_geometry_light.h"

#include "sparks/core/core.h"
#include "sparks/core/geometry.h"
#include "sparks/core/scene.h"
#include "sparks/core/surface.h"

namespace sparks {

EntityGeometryLight::EntityGeometryLight(Core *core,
                                         Geometry *geometry,
                                         const glm::vec3 &emission,
                                         bool two_sided,
                                         bool block_ray,
                                         const glm::mat4x3 &transform)
    : Entity(core),
      surface_light_(core, emission, two_sided, block_ray),
      light_geom_surf_(core, geometry, &surface_light_, transform),
      geometry_(geometry),
      emission(emission),
      two_sided(two_sided),
      block_ray(block_ray),
      transformation_(transform) {
}

void EntityGeometryLight::Update(Scene *scene) {
  surface_light_.emission = emission;
  surface_light_.two_sided = two_sided;
  surface_light_.block_ray = block_ray;
  light_geom_surf_.SamplerShader();
  int32_t light_index = scene->RegisterLight(&light_geom_surf_);
  int32_t instance_index = scene->RegisterInstance(scene->RegisterGeometry(geometry_), transformation_,
                                                   scene->RegisterSurface(&surface_light_), light_index);
  scene->LightCustomIndex(light_index) = instance_index;
}

}  // namespace sparks
