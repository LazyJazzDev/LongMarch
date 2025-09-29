#include "sparkium/material/material_light.h"

#include "sparkium/core/core.h"

namespace sparkium {

MaterialLight::MaterialLight(Core *core, const glm::vec3 &emission, bool two_sided, bool block_ray)
    : Material(core), emission(emission), two_sided(two_sided), block_ray(block_ray) {
}

}  // namespace sparkium
