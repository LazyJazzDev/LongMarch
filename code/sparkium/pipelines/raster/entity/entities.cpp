#include "sparkium/pipelines/raster/entity/entities.h"

namespace sparkium::raster {

Entity *DedicatedCast(sparkium::Entity *entity) {
  DEDICATED_CAST(entity, sparkium::EntityGeometryMaterial, EntityGeometryMaterial)
  DEDICATED_CAST(entity, sparkium::EntityPointLight, EntityPointLight)
  return nullptr;
}

}  // namespace sparkium::raster
