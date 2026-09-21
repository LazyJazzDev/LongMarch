#include "sparkium/backend/cuda/path_tracing/entity/entity_point_light.h"

#include "sparkium/backend/cuda/path_tracing/core/core.h"
#include "sparkium/backend/cuda/path_tracing/core/scene.h"

namespace sparkium::cuda_tracing {

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

}  // namespace sparkium::cuda_tracing
