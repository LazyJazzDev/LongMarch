#include "cao_di/math/math.h"

namespace CD {

void PyBindMath(pybind11::module_ &m) {
  pybind11::class_<MeshSDF> mesh_sdf(m, "MeshSDF");
  mesh_sdf.def(pybind11::init([](const std::vector<Vector3<float>> &vertices, const std::vector<uint32_t> &indices) {
    return MeshSDF{VertexBufferView{vertices.data()}, vertices.size(), indices.data(), indices.size()};
  }));
  mesh_sdf.def("__repr__", [](const MeshSDF &self) {
    return pybind11::str("MeshSDF(\n vts={}\n inds={}\n edge_inds={}\n)")
        .format(self.GetVertices(), self.GetTriangleIndices(), self.GetEdgeIndices());
  });

  m.def("rotation", [](const Vector3<float> &rot_vec) -> Matrix3<float> {
    // return the rotation matrix related to the rotation vector rot_vec
    if (rot_vec.norm() < 1e-6f) {
      return Matrix3<float>::Identity();
    }
    return Eigen::AngleAxis<float>(rot_vec.norm(), rot_vec.normalized()).toRotationMatrix();
  });
}

}  // namespace CD
