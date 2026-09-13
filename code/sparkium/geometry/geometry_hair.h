#pragma once

#include "sparkium/geometry/geometry_mesh.h"

namespace sparkium {

// Curve strands are stored compactly by scene files and expanded to tapered
// polygonal tubes for the current mesh-based rendering backends.
class GeometryHair : public GeometryMesh {
 public:
  GeometryHair(Core *core,
               const std::vector<Vector3<float>> &points,
               const std::vector<float> &radii,
               const std::vector<uint32_t> &strand_offsets,
               int radial_segments = 3);
};

}  // namespace sparkium
