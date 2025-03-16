#include "snow_mount/visualizer/visualizer_mesh.h"

#include "snow_mount/visualizer/visualizer_core.h"

namespace snow_mount::visualizer {

Mesh::Mesh(const std::shared_ptr<Core> &core) : core_(core) {
  auto graphics_core = core->GraphicsCore();
  graphics_core->CreateBuffer(4096, graphics::BufferType::BUFFER_TYPE_STATIC, &vertex_buffer_);
  graphics_core->CreateBuffer(4096, graphics::BufferType::BUFFER_TYPE_STATIC, &index_buffer_);
  num_vertices_ = 0;
  num_indices_ = 0;
}

void Mesh::SetVertices(const Vertex *vertices) {
  SetVertices(vertices, num_vertices_);
}

void Mesh::SetIndices(const uint32_t *indices) {
  SetIndices(indices, num_indices_);
}

void Mesh::SetVertices(const Vertex *vertices, int num_vertex) {
  if (vertex_buffer_->Size() < num_vertex * sizeof(Vertex)) {
    vertex_buffer_->Resize(num_vertex * sizeof(Vertex));
  }
  vertex_buffer_->UploadData(vertices, num_vertex * sizeof(Vertex));
  num_vertices_ = num_vertex;
}

void Mesh::SetIndices(const uint32_t *indices, int num_indices) {
  if (index_buffer_->Size() < num_indices * sizeof(uint32_t)) {
    index_buffer_->Resize(num_indices * sizeof(uint32_t));
  }
  index_buffer_->UploadData(indices, num_indices * sizeof(uint32_t));
  num_indices_ = num_indices;
}

int Mesh::GetVertexCount() const {
  return num_vertices_;
}

int Mesh::GetIndexCount() const {
  return num_indices_;
}

graphics::Buffer *Mesh::GetVertexBuffer() const {
  return vertex_buffer_.get();
}

graphics::Buffer *Mesh::GetIndexBuffer() const {
  return index_buffer_.get();
}

void Mesh::PyBind(pybind11::module_ &m) {
  pybind11::class_<Mesh, std::shared_ptr<Mesh>> mesh(m, "Mesh");
  mesh.def("set_vertices", [](Mesh &mesh, pybind11::array_t<float> vertices) {
    auto r = vertices.unchecked<2>();
    std::vector<Vertex> vertex_data;
    if (r.shape(1) == 3) {
      vertex_data.resize(r.shape(0));
      for (size_t i = 0; i < r.shape(0); i++) {
        vertex_data[i].position = {r(i, 0), r(i, 1), r(i, 2)};
        vertex_data[i].normal = {};
        vertex_data[i].tex_coord = {};
        vertex_data[i].color = {1.0, 1.0, 1.0, 1.0};
      }
    } else if (r.shape(1) == 12) {
      vertex_data.resize(r.shape(0));
      for (size_t i = 0; i < r.shape(0); i++) {
        vertex_data[i].position = {r(i, 0), r(i, 1), r(i, 2)};
        vertex_data[i].normal = {r(i, 3), r(i, 4), r(i, 5)};
        vertex_data[i].tex_coord = {r(i, 6), r(i, 7)};
        vertex_data[i].color = {r(i, 8), r(i, 9), r(i, 10), r(i, 11)};
      }
    } else {
      throw std::runtime_error("Number of dimensions must be 3 or 12");
    }
    mesh.SetVertices(vertex_data.data(), vertex_data.size());
  });

  mesh.def("set_indices", [](Mesh &mesh, pybind11::array_t<uint32_t> indices) {
    auto r = indices.unchecked<1>();
    std::vector<uint32_t> index_data(r.shape(0));
    for (size_t i = 0; i < r.shape(0); i++) {
      index_data[i] = r(i);
    }
    mesh.SetIndices(index_data.data(), index_data.size());
  });

  mesh.def("get_vertex_count", &Mesh::GetVertexCount);
  mesh.def("get_index_count", &Mesh::GetIndexCount);
}

}  // namespace snow_mount::visualizer
