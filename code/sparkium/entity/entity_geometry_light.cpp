#include "sparkium/entity/entity_geometry_light.h"

#include "sparkium/core/core.h"

namespace sparkium {

EntityGeometryLight::EntityGeometryLight(Core *core,
                                         Geometry *geometry,
                                         const glm::vec3 &emission,
                                         bool two_sided,
                                         bool block_ray,
                                         const glm::mat4x3 &transform)
    : Entity(core),
      geometry_(geometry),
      emission(emission),
      two_sided(two_sided),
      block_ray(block_ray),
      transform(transform) {
}

}  // namespace sparkium
