#include "sparkium/geometry/geometry_hair.h"

#include <cmath>
#include <stdexcept>

namespace sparkium {
namespace {
Mesh<float> BuildHairMesh(const std::vector<Vector3<float>> &points,
                          const std::vector<float> &radii,
                          const std::vector<uint32_t> &strand_offsets,
                          int radial_segments) {
  if (points.size() != radii.size() || strand_offsets.size() < 2 || strand_offsets.front() != 0 ||
      strand_offsets.back() != points.size() || radial_segments < 3)
    throw std::runtime_error("invalid hair geometry");

  std::vector<Vector3<float>> positions;
  std::vector<Vector3<float>> normals;
  std::vector<Vector2<float>> tex_coords;
  std::vector<uint32_t> indices;
  positions.reserve(points.size() * radial_segments);
  normals.reserve(points.size() * radial_segments);
  tex_coords.reserve(points.size() * radial_segments);

  for (size_t strand = 0; strand + 1 < strand_offsets.size(); ++strand) {
    const uint32_t begin = strand_offsets[strand];
    const uint32_t end = strand_offsets[strand + 1];
    if (end <= begin + 1 || end > points.size())
      throw std::runtime_error("invalid hair strand offsets");
    const uint32_t vertex_base = static_cast<uint32_t>(positions.size());
    for (uint32_t i = begin; i < end; ++i) {
      Vector3<float> tangent;
      if (i == begin)
        tangent = points[i + 1] - points[i];
      else if (i + 1 == end)
        tangent = points[i] - points[i - 1];
      else
        tangent = points[i + 1] - points[i - 1];
      tangent.normalize();
      Vector3<float> axis =
          std::abs(tangent.z()) < 0.9f ? Vector3<float>{0.0f, 0.0f, 1.0f} : Vector3<float>{0.0f, 1.0f, 0.0f};
      Vector3<float> side = tangent.cross(axis).normalized();
      Vector3<float> up = side.cross(tangent).normalized();
      const float v = float(i - begin) / float(end - begin - 1);
      for (int j = 0; j < radial_segments; ++j) {
        const float angle = 2.0f * grassland::PI<float>() * float(j) / float(radial_segments);
        const Vector3<float> normal = std::cos(angle) * side + std::sin(angle) * up;
        positions.push_back(points[i] + normal * radii[i]);
        normals.push_back(normal);
        tex_coords.push_back({float(j) / float(radial_segments), v});
      }
    }
    const uint32_t rings = end - begin;
    for (uint32_t ring = 0; ring + 1 < rings; ++ring) {
      for (int j = 0; j < radial_segments; ++j) {
        const uint32_t a = vertex_base + ring * radial_segments + j;
        const uint32_t b = vertex_base + ring * radial_segments + (j + 1) % radial_segments;
        const uint32_t c = a + radial_segments;
        const uint32_t d = b + radial_segments;
        indices.insert(indices.end(), {a, c, b, b, c, d});
      }
    }
  }
  return Mesh<float>(positions.size(), indices.size(), indices.data(), positions.data(), normals.data(),
                     tex_coords.data());
}
}  // namespace

GeometryHair::GeometryHair(Core *core,
                           const std::vector<Vector3<float>> &points,
                           const std::vector<float> &radii,
                           const std::vector<uint32_t> &strand_offsets,
                           int radial_segments)
    : GeometryMesh(core, BuildHairMesh(points, radii, strand_offsets, radial_segments)) {
}

}  // namespace sparkium
