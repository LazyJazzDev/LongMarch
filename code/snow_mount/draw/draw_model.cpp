#include "snow_mount/draw/draw_model.h"

#include "snow_mount/draw/draw_core.h"

namespace XS::draw {

Model::Model(Core *core) : core_(core) {
  core_->GraphicsCore()->CreateBuffer(sizeof(Vertex), graphics::BUFFER_TYPE_DYNAMIC, &vertex_buffer_);
  core_->GraphicsCore()->CreateBuffer(sizeof(uint32_t), graphics::BUFFER_TYPE_DYNAMIC, &index_buffer_);
}

void Model::SetModelData(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices) {
  if (vertices.size() * sizeof(Vertex) > vertex_buffer_->Size()) {
    vertex_buffer_->Resize(vertices.size() * sizeof(Vertex));
  }
  if (indices.size() * sizeof(uint32_t) > index_buffer_->Size()) {
    index_buffer_->Resize(indices.size() * sizeof(uint32_t));
  }
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(Vertex));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));
  index_count_ = indices.size();
}

}  // namespace XS::draw
