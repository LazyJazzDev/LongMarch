#include "sparkium/entity/entity_area_light.h"

#include <glm/ext/matrix_transform.hpp>

namespace sparkium {
EntityAreaLight::EntityAreaLight(Core *core,
                                 const glm::vec3 &emission,
                                 float size,
                                 const glm::vec3 &position,
                                 const glm::vec3 &direction,
                                 const glm::vec3 &up)
    : Entity(core), emission(emission), position(position), direction(direction), size(size), up(up) {
}

}  // namespace sparkium
