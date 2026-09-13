#include "sparkium/entity/entity_point_light.h"

#include "sparkium/core/core.h"

namespace sparkium {

EntityPointLight::EntityPointLight(Core *core,
                                   const glm::vec3 &position,
                                   const glm::vec3 &color,
                                   float strength,
                                   float radius,
                                   bool soft_falloff,
                                   float sampling_weight)
    : Entity(core),
      position(position),
      color(color),
      strength(strength),
      radius(radius),
      soft_falloff(soft_falloff),
      sampling_weight(sampling_weight) {
}

}  // namespace sparkium
