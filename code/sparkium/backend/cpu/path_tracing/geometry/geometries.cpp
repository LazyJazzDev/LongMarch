#include "sparkium/backend/cpu/path_tracing/geometry/geometries.h"

namespace sparkium::cpu_tracing {

Geometry *DedicatedCast(sparkium::Geometry *geometry) {
  DEDICATED_CAST(geometry, sparkium::GeometryMesh, GeometryMesh);
  return nullptr;
}

}  // namespace sparkium::cpu_tracing
