#include "sparkium/material/material_light.h"

#include "sparkium/core/core.h"

namespace sparkium {

MaterialLight::MaterialLight(Core *core,
                             const glm::vec3 &emission,
                             bool two_sided,
                             bool block_ray,
                             bool camera_visible,
                             float falloff_distance)
    : Material(core),
      emission(emission),
      two_sided(two_sided),
      block_ray(block_ray),
      camera_visible(camera_visible),
      falloff_distance(falloff_distance) {
}

}  // namespace sparkium
