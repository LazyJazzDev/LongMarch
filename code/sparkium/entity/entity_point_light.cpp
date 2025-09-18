#include "sparkium/entity/entity_point_light.h"

#include "sparkium/core/core.h"
#include "sparkium/core/scene.h"

namespace sparkium {

EntityPointLight::EntityPointLight(Core *core, const glm::vec3 &position, const glm::vec3 &color, float strength)
    : Entity(core), light_point_(core, position, color, strength) {
}

void EntityPointLight::Update(Scene *scene) {
  scene->RegisterLight(&light_point_);
}

}  // namespace sparkium
