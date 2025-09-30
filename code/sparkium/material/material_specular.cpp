#include "sparkium/material/material_specular.h"

#include "sparkium/core/core.h"

namespace sparkium {

MaterialSpecular::MaterialSpecular(Core *core, const glm::vec3 &base_color) : Material(core), base_color(base_color) {
}

}  // namespace sparkium
