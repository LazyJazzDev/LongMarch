#pragma once
#include "grassland/math/math_util.h"

namespace grassland {

struct MeshSDFRef {
  const Vector3<float> *x;
  const uint32_t *triangle_indices;
  const uint32_t *edge_indices;
  const uint8_t *edge_inside;
  const uint8_t *point_inside;
  int num_triangles;
  int num_edges;
  int num_points;

  Matrix3<float> rotation;
  Vector3<float> translation;

  LM_DEVICE_FUNC Vector3<float> GetPosition(int index) const;
  LM_DEVICE_FUNC void SDF(const Vector3<float> &position,
                          float *sdf,
                          Vector3<float> *jacobian,
                          Matrix3<float> *hessian) const;
};

class MeshSDF {
 public:
  MeshSDF(VertexBufferView vertex_buffer_view, size_t num_vertex, uint32_t *indices, size_t num_indices);

  operator MeshSDFRef() const;

 private:
  std::vector<Vector3<float>> x_;
  std::vector<uint32_t> triangle_indices_;
  std::vector<uint32_t> edge_indices_;
  std::vector<uint8_t> edge_inside_;
  std::vector<uint8_t> point_inside_;
};

}  // namespace grassland
