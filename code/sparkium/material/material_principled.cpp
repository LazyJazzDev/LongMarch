#include "material_principled.h"

#include "sparkium/core/core.h"
#include "sparkium/core/scene.h"
#include "sparkium/material/material_principled.h"

namespace sparkium {

MaterialPrincipled::MaterialPrincipled(Core *core, const glm::vec3 &base_color) : Material(core), info{base_color} {
}

}  // namespace sparkium
