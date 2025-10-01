#include "sparkium/pipelines/raster/entity/entity_geometry_material.h"

#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/geometry/geometries.h"
#include "sparkium/pipelines/raster/material/materials.h"

namespace sparkium::raster {

EntityGeometryMaterial::EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity)
    : entity_(entity),
      Entity(DedicatedCast(entity.GetCore())),
      geometry_(DedicatedCast(entity_.GetGeometry())),
      material_(DedicatedCast(entity_.GetMaterial())) {
}

void EntityGeometryMaterial::Update(Scene *scene) {
}

}  // namespace sparkium::raster
