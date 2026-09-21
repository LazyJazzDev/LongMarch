#include "sparkium/backend/cpu/path_tracing/entity/entities.h"

namespace sparkium::cpu_tracing {

Entity *DedicatedCast(sparkium::Entity *entity) {
  DEDICATED_CAST(entity, sparkium::EntityGeometryMaterial, EntityGeometryMaterial)
  DEDICATED_CAST(entity, sparkium::EntityPointLight, EntityPointLight)
  return nullptr;
}

}  // namespace sparkium::cpu_tracing
