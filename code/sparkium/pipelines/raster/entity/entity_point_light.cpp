#include "sparkium/pipelines/raster/entity/entity_point_light.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

EntityPointLight::EntityPointLight(sparkium::EntityPointLight &entity)
    : entity_(entity), Entity(DedicatedCast(entity.GetCore())) {
}

void EntityPointLight::Update(Scene *scene) {
}

}  // namespace sparkium::raster
