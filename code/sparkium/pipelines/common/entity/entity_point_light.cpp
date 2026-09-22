#include "sparkium/pipelines/common/entity/entity_point_light.h"

#include "sparkium/pipelines/common/core/core.h"
#include "sparkium/pipelines/common/core/scene.h"

namespace sparkium::render_shared {

EntityPointLight::EntityPointLight(sparkium::EntityPointLight &entity)
    : entity_(entity),
      Entity(DedicatedCast(entity.GetCore())),
      light_point_(core_,
                   entity.position,
                   entity.color,
                   entity.strength,
                   entity.radius,
                   entity.soft_falloff,
                   entity.sampling_weight) {
}

void EntityPointLight::Update(Scene *scene) {
  scene->RegisterLight(&light_point_);
}

}  // namespace sparkium::render_shared
