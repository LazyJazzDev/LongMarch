#include "snowberg/visualizer/visualizer_mesh.h"

#include "snowberg/visualizer/visualizer_core.h"

namespace snowberg::visualizer {

Mesh::Mesh(const std::shared_ptr<Core> &core) : core_(core) {
  auto graphics_core = core->GraphicsCore();
  graphics_core->CreateBuffer(4096, graphics::BufferType::BUFFER_TYPE_STATIC, &vertex_buffer_);
  graphics_core->CreateBuffer(4096, graphics::BufferType::BUFFER_TYPE_STATIC, &index_buffer_);
  num_vertices_ = 0;
  num_indices_ = 0;
}

std::shared_ptr<Core> Mesh::GetCore() const {
  return core_;
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

int Mesh::VertexCount() const {
  return num_vertices_;
}

int Mesh::IndexCount() const {
  return num_indices_;
}

graphics::Buffer *Mesh::GetVertexBuffer() const {
  return vertex_buffer_.get();
}

graphics::Buffer *Mesh::GetIndexBuffer() const {
  return index_buffer_.get();
}

}  // namespace snowberg::visualizer
