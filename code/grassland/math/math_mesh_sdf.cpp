#include "grassland/math/math_mesh_sdf.h"

#include <grassland/physics/diff_kernel/dk_geometry_sdf.h>

#include "math_basics.h"
#include "math_static_collision.h"

namespace grassland {

LM_DEVICE_FUNC void MeshSDFRef::SDF(const Vector3<float> &position,
                                    const Matrix3<float> &R,
                                    const Vector3<float> &t,
                                    float *sdf,
                                    Vector3<float> *jacobian,
                                    Matrix3<float> *hessian) const {
  float u, v;
  float local_sdf = std::numeric_limits<float>::max();
  Vector3<float> local_jacobian = Vector3<float>::Zero();
  Matrix3<float> local_hessian = Matrix3<float>::Zero();
  for (int i = 0; i < num_triangles; i++) {
    int a = triangle_indices[i * 3 + 0];
    int b = triangle_indices[i * 3 + 1];
    int c = triangle_indices[i * 3 + 2];
    Vector3<float> pa = R * x[a] + t;
    Vector3<float> pb = R * x[b] + t;
    Vector3<float> pc = R * x[c] + t;
    Vector3<float> n = (pb - pa).cross(pc - pa);
    float t = DistancePointPlane(position, pa, pb, pc, u, v);
    if (u >= 0 && v >= 0 && u + v <= 1) {
      if (t < fabs(local_sdf)) {
        if ((position - pa).dot(n) < 0) {
          t = -t;
        }
        local_sdf = t;
        n.normalize();
        local_jacobian = n;
      }
    }
  }
  for (int i = 0; i < num_edges; i++) {
    int a = edge_indices[i * 2 + 0];
    int b = edge_indices[i * 2 + 1];
    Vector3<float> pa = R * x[a] + t;
    Vector3<float> pb = R * x[b] + t;
    float t = DistancePointLine(position, pa, pb, u);
    if (0 < u && u < 1) {
      if (t < fabs(local_sdf)) {
        LineSDF<float> line_sdf;
        line_sdf.origin = pa;
        line_sdf.direction = (pb - pa).normalized();
        local_sdf = t;
        local_jacobian = line_sdf.Jacobian(position);
        local_hessian = line_sdf.Hessian(position).m[0];
        if (edge_inside[i]) {
          local_sdf = -local_sdf;
          local_jacobian = -local_jacobian;
          local_hessian = -local_hessian;
        }
      }
    }
  }
  for (int i = 0; i < num_points; i++) {
    PointSDF<float> point_sdf;
    point_sdf.position = R * x[i] + t;
    float t = point_sdf(position).value();
    if (t < fabs(local_sdf)) {
      local_sdf = t;
      local_jacobian = point_sdf.Jacobian(position);
      local_hessian = point_sdf.Hessian(position).m[0];
      if (point_inside[i]) {
        local_sdf = -local_sdf;
        local_jacobian = -local_jacobian;
        local_hessian = -local_hessian;
      }
    }
  }
  *sdf = local_sdf;
  if (jacobian) {
    *jacobian = local_jacobian;
  }
  if (hessian) {
    *hessian = local_hessian;
  }
}

MeshSDF::MeshSDF(VertexBufferView vertex_buffer_view, size_t num_vertex, const uint32_t *indices, size_t num_indices) {
  x_.resize(num_vertex);
  triangle_indices_.resize(num_indices);
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> map_third_vertex;
  for (int i = 0; i < num_vertex; i++) {
    x_[i] = vertex_buffer_view.Get<Vector3<float>>(i);
  }
  std::memcpy(triangle_indices_.data(), indices, num_indices * sizeof(uint32_t));

  for (int i = 0; i < triangle_indices_.size(); i += 3) {
    int u = triangle_indices_[i];
    int v = triangle_indices_[i + 1];
    int w = triangle_indices_[i + 2];
    map_third_vertex[{u, v}] = w;
    map_third_vertex[{v, w}] = u;
    map_third_vertex[{w, u}] = v;
  }

  edge_inside_.reserve(map_third_vertex.size() / 2);
  edge_indices_.reserve(map_third_vertex.size());

  std::vector<Vector3<float>> edge_mean(num_vertex, Vector3<float>::Zero());
  for (auto [uv, w] : map_third_vertex) {
    auto u = uv.first;
    auto v = uv.second;
    Vector3<float> pu = x_[u];
    Vector3<float> pv = x_[v];
    edge_mean[u] += pv - pu;

    if (u < v) {
      if (map_third_vertex.count({v, u}) > 0) {
        int w_other = map_third_vertex[{v, u}];
        Vector3<float> pw = x_[w];
        Vector3<float> pw_other = x_[w_other];
        Vector3<float> n = (pv - pu).cross(pw - pu);
        Vector3<float> n_other = (pu - pv).cross(pw_other - pv);
        Vector3<float> dw = pw_other - pw;
        edge_indices_.push_back(u);
        edge_indices_.push_back(v);
        edge_inside_.push_back(dw.dot(n) > 0);
      }
    }
  }

  std::vector<float> solid_angle(num_vertex, 0);

  for (auto [uv, w] : map_third_vertex) {
    auto u = uv.first;
    auto v = uv.second;
    solid_angle[u] += SolidAngle<float>(edge_mean[u], x_[v] - x_[u], x_[w] - x_[u]);
  }

  point_inside_.resize(num_vertex);
  for (int i = 0; i < num_vertex; i++) {
    point_inside_[i] = (solid_angle[i] > 0);
  }
}

MeshSDF::operator MeshSDFRef() const {
  MeshSDFRef mesh_sdf;
  mesh_sdf.num_triangles = triangle_indices_.size() / 3;
  mesh_sdf.num_edges = edge_indices_.size() / 2;
  mesh_sdf.num_points = x_.size();
  mesh_sdf.x = x_.data();
  mesh_sdf.triangle_indices = triangle_indices_.data();
  mesh_sdf.edge_indices = edge_indices_.data();
  mesh_sdf.edge_inside = edge_inside_.data();
  mesh_sdf.point_inside = point_inside_.data();
  return mesh_sdf;
}

#if defined(__CUDACC__)
MeshSDFDevice::MeshSDFDevice(const MeshSDF &mesh_sdf) {
  x_ = mesh_sdf.x_;
  triangle_indices_ = mesh_sdf.triangle_indices_;
  edge_indices_ = mesh_sdf.edge_indices_;
  edge_inside_ = mesh_sdf.edge_inside_;
  point_inside_ = mesh_sdf.point_inside_;
}

MeshSDFDevice::operator MeshSDFRef() const {
  MeshSDFRef mesh_sdf;
  mesh_sdf.num_triangles = triangle_indices_.size() / 3;
  mesh_sdf.num_edges = edge_indices_.size() / 2;
  mesh_sdf.num_points = x_.size();
  mesh_sdf.x = x_.data().get();
  mesh_sdf.triangle_indices = triangle_indices_.data().get();
  mesh_sdf.edge_indices = edge_indices_.data().get();
  mesh_sdf.edge_inside = edge_inside_.data().get();
  mesh_sdf.point_inside = point_inside_.data().get();
  return mesh_sdf;
}

MeshSDFDevice::operator MeshSDF() const {
  MeshSDF mesh_sdf;
  thrust::copy(x_.begin(), x_.end(), std::back_inserter(mesh_sdf.x_));
  thrust::copy(triangle_indices_.begin(), triangle_indices_.end(), std::back_inserter(mesh_sdf.triangle_indices_));
  thrust::copy(edge_indices_.begin(), edge_indices_.end(), std::back_inserter(mesh_sdf.edge_indices_));
  thrust::copy(edge_inside_.begin(), edge_inside_.end(), std::back_inserter(mesh_sdf.edge_inside_));
  thrust::copy(point_inside_.begin(), point_inside_.end(), std::back_inserter(mesh_sdf.point_inside_));
  return mesh_sdf;
}
#endif

}  // namespace grassland
