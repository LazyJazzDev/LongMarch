#include "sparkium/pipelines/realtime/geometry/geometries.h"

namespace sparkium::realtime {

Geometry *DedicatedCast(sparkium::Geometry *geometry) {
  DEDICATED_CAST(geometry, sparkium::GeometryMesh, GeometryMesh);
  return nullptr;
}

}  // namespace sparkium::realtime
