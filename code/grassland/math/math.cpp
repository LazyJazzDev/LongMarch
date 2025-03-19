#include "grassland/math/math.h"

namespace grassland {

void PyBindMath(pybind11::module_ &m) {
  pybind11::class_<MeshSDF> mesh_sdf(m, "MeshSDF");
  mesh_sdf.def(pybind11::init([](const std::vector<Vector3<float>> &vertices, const std::vector<uint32_t> &indices) {
    return MeshSDF{VertexBufferView{vertices.data()}, vertices.size(), indices.data(), indices.size()};
  }));
  mesh_sdf.def("__repr__", [](const MeshSDF &self) {
    return pybind11::str("MeshSDF(\n vts={}\n inds={}\n edge_inds={}\n)")
        .format(self.GetVertices(), self.GetTriangleIndices(), self.GetEdgeIndices());
  });
}

}  // namespace grassland
