#include "sparkium/pipelines/raster/geometry/geometries.h"

namespace sparkium::raster {

Geometry *DedicatedCast(sparkium::Geometry *geometry) {
  DEDICATED_CAST(geometry, sparkium::GeometryMesh, GeometryMesh);
  return nullptr;
}

}  // namespace sparkium::raster
