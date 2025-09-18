#include "sparkium/entity/entity_geometry_light.h"

#include "sparkium/core/core.h"
#include "sparkium/core/geometry.h"
#include "sparkium/core/material.h"
#include "sparkium/core/scene.h"

namespace sparkium {

EntityGeometryLight::EntityGeometryLight(Core *core,
                                         Geometry *geometry,
                                         const glm::vec3 &emission,
                                         bool two_sided,
                                         bool block_ray,
                                         const glm::mat4x3 &transform)
    : Entity(core),
      material_light_(core, emission, two_sided, block_ray),
      geometry_(geometry),
      emission(emission),
      two_sided(two_sided),
      block_ray(block_ray),
      transform(transform) {
  entity_ = std::make_unique<EntityGeometryMaterial>(core_, geometry, &material_light_, transform);
}

void EntityGeometryLight::Update(Scene *scene) {
  material_light_.emission = emission;
  material_light_.two_sided = two_sided;
  material_light_.block_ray = block_ray;
  entity_->SetTransformation(transform);
  entity_->Update(scene);
}

}  // namespace sparkium
