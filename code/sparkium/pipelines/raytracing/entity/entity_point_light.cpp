#include "sparkium/pipelines/raytracing/entity/entity_point_light.h"

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

EntityPointLight::EntityPointLight(sparkium::EntityPointLight &entity)
    : entity_(entity),
      Entity(DedicatedCast(entity.GetCore())),
      light_point_(core_, entity.position, entity.color, entity.strength) {
}

void EntityPointLight::Update(Scene *scene) {
  scene->RegisterLight(&light_point_);
}

}  // namespace sparkium::raytracing
