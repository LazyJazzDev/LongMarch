#include "sparkium/entity/entity_point_light.h"

#include "sparkium/core/core.h"

namespace sparkium {

EntityPointLight::EntityPointLight(Core *core, const glm::vec3 &position, const glm::vec3 &color, float strength)
    : Entity(core), position(position), color(color), strength(strength) {
}

}  // namespace sparkium
