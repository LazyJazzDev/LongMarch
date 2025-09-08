#pragma once
#include "grassland/math/math_util.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif

namespace CD {

struct MeshSDFRef {
  const Vector3<float> *x;
  const uint32_t *triangle_indices;
  const uint32_t *edge_indices;
  const uint8_t *edge_inside;
  const uint8_t *point_inside;
  int num_triangles;
  int num_edges;
  int num_points;

  LM_DEVICE_FUNC void SDF(const Vector3<float> &position,
                          const Matrix3<float> &R,
                          const Vector3<float> &t,
                          float *sdf,
                          Vector3<float> *jacobian,
                          Matrix3<float> *hessian) const;
};

class MeshSDF {
 public:
  MeshSDF() = default;
  MeshSDF(VertexBufferView vertex_buffer_view, size_t num_vertex, const uint32_t *indices, size_t num_indices);

  operator MeshSDFRef() const;

  const std::vector<Vector3<float>> &GetVertices() const {
    return x_;
  }

  const std::vector<uint32_t> &GetTriangleIndices() const {
    return triangle_indices_;
  }

  const std::vector<uint32_t> &GetEdgeIndices() const {
    return edge_indices_;
  }

  const std::vector<uint8_t> &GetEdgeInside() const {
    return edge_inside_;
  }

  const std::vector<uint8_t> &GetPointInside() const {
    return point_inside_;
  }

 private:
  friend class MeshSDFDevice;
  std::vector<Vector3<float>> x_;
  std::vector<uint32_t> triangle_indices_;
  std::vector<uint32_t> edge_indices_;
  std::vector<uint8_t> edge_inside_;
  std::vector<uint8_t> point_inside_;
};

#if defined(__CUDACC__)
class MeshSDFDevice {
 public:
  MeshSDFDevice() = default;
  MeshSDFDevice(const MeshSDF &mesh_sdf);

  operator MeshSDFRef() const;
  operator MeshSDF() const;

 private:
  thrust::device_vector<Vector3<float>> x_;
  thrust::device_vector<uint32_t> triangle_indices_;
  thrust::device_vector<uint32_t> edge_indices_;
  thrust::device_vector<uint8_t> edge_inside_;
  thrust::device_vector<uint8_t> point_inside_;
};
#endif

}  // namespace CD
