#include "sparkium/pipelines/common/geometry/geometries.h"

namespace sparkium::render_shared {

Geometry *DedicatedCast(sparkium::Geometry *geometry) {
  DEDICATED_CAST(geometry, sparkium::GeometryMesh, GeometryMesh);
  return nullptr;
}

}  // namespace sparkium::render_shared
