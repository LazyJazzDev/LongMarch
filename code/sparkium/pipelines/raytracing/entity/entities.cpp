#include "sparkium/pipelines/raytracing/entity/entities.h"

namespace sparkium::raytracing {

Entity *DedicatedCast(sparkium::Entity *entity) {
  DEDICATED_CAST(entity, sparkium::EntityGeometryMaterial, EntityGeometryMaterial)
  DEDICATED_CAST(entity, sparkium::EntityPointLight, EntityPointLight)
  return nullptr;
}

}  // namespace sparkium::raytracing
