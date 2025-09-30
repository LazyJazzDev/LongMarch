#include "sparkium/material/material_lambertian.h"

#include "sparkium/core/core.h"

namespace sparkium {

MaterialLambertian::MaterialLambertian(Core *core, const glm::vec3 &base_color, const glm::vec3 &emission)
    : Material(core), base_color(base_color), emission(emission) {
}

}  // namespace sparkium
