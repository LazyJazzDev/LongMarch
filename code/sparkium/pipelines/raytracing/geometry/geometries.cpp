#include "sparkium/pipelines/raytracing/geometry/geometries.h"

namespace sparkium::raytracing {

Geometry *DedicatedCast(sparkium::Geometry *geometry) {
  DEDICATED_CAST(geometry, sparkium::GeometryMesh, GeometryMesh);
  return nullptr;
}

}  // namespace sparkium::raytracing
