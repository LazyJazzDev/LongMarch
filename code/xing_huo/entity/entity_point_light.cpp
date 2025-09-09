#include "xing_huo/entity/entity_point_light.h"

#include "xing_huo/core/core.h"
#include "xing_huo/core/scene.h"

namespace XH {

EntityPointLight::EntityPointLight(Core *core, const glm::vec3 &position, const glm::vec3 &color, float strength)
    : Entity(core), light_point_(core, position, color, strength) {
}

void EntityPointLight::Update(Scene *scene) {
  scene->RegisterLight(&light_point_);
}

}  // namespace XH
