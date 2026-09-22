#include "sparkium/pipelines/common/entity/entities.h"

namespace sparkium::render_shared {

Entity *DedicatedCast(sparkium::Entity *entity) {
  DEDICATED_CAST(entity, sparkium::EntityGeometryMaterial, EntityGeometryMaterial)
  DEDICATED_CAST(entity, sparkium::EntityPointLight, EntityPointLight)
  return nullptr;
}

}  // namespace sparkium::render_shared
