#include "sparkium/backend/cuda/path_tracing/geometry/geometries.h"

namespace sparkium::cuda_tracing {

Geometry *DedicatedCast(sparkium::Geometry *geometry) {
  DEDICATED_CAST(geometry, sparkium::GeometryMesh, GeometryMesh);
  return nullptr;
}

}  // namespace sparkium::cuda_tracing
