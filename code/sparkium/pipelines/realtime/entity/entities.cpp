#include "sparkium/pipelines/realtime/entity/entities.h"

namespace sparkium::realtime {

Entity *DedicatedCast(sparkium::Entity *entity) {
  DEDICATED_CAST(entity, sparkium::EntityGeometryMaterial, EntityGeometryMaterial)
  DEDICATED_CAST(entity, sparkium::EntityPointLight, EntityPointLight)
  return nullptr;
}

}  // namespace sparkium::realtime
