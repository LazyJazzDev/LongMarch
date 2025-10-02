#include "sparkium/geometry/geometry_mesh.h"

#include "sparkium/core/core.h"

namespace sparkium {

GeometryMesh::GeometryMesh(Core *core, const Mesh<float> &mesh) : Geometry(core) {
  std::vector<uint8_t> data;
  auto write_data = [&](const void *data_ptr, size_t size) {
    data.insert(data.end(), static_cast<const uint8_t *>(data_ptr), static_cast<const uint8_t *>(data_ptr) + size);
  };
  write_data(&header_, sizeof(header_));

  Mesh<float> mesh_copy;
  const Mesh<float> *mesh_ptr = &mesh;
  if (mesh.Normals() && mesh.TexCoords() && !mesh.Tangents()) {
    mesh_copy = mesh;
    mesh_copy.GenerateTangents();
    mesh_ptr = &mesh_copy;
  }

  header_.num_indices = mesh_ptr->NumIndices();
  header_.num_vertices = mesh_ptr->NumVertices();

  header_.index_offset = data.size();
  write_data(mesh_ptr->Indices(), mesh_ptr->NumIndices() * sizeof(uint32_t));
  header_.position_offset = data.size();
  header_.position_stride = sizeof(float) * 3;
  write_data(mesh_ptr->Positions(), mesh_ptr->NumVertices() * sizeof(float) * 3);

  if (mesh_ptr->Normals()) {
    header_.normal_offset = data.size();
    header_.normal_stride = sizeof(float) * 3;
    write_data(mesh_ptr->Normals(), mesh_ptr->NumVertices() * sizeof(float) * 3);
  }

  if (mesh_ptr->TexCoords()) {
    header_.tex_coord_offset = data.size();
    header_.tex_coord_stride = sizeof(float) * 2;
    write_data(mesh_ptr->TexCoords(), mesh_ptr->NumVertices() * sizeof(float) * 2);
  }

  if (mesh_ptr->Tangents()) {
    header_.tangent_offset = data.size();
    header_.tangent_stride = sizeof(float) * 3;
    write_data(mesh_ptr->Tangents(), mesh_ptr->NumVertices() * sizeof(float) * 3);
    header_.signal_offset = data.size();
    header_.signal_stride = sizeof(float);
    write_data(mesh_ptr->Signals(), mesh_ptr->NumVertices() * sizeof(float));
  }

  std::memcpy(data.data(), &header_, sizeof(header_));

  core_->GraphicsCore()->CreateBuffer(data.size(), graphics::BUFFER_TYPE_STATIC, &geometry_buffer_);
  geometry_buffer_->UploadData(data.data(), data.size());
  primitive_count_ = header_.num_indices / 3;
}

int GeometryMesh::PrimitiveCount() {
  return primitive_count_;
}

graphics::Buffer *GeometryMesh::GetBuffer() const {
  return geometry_buffer_.get();
}

const GeometryMesh::Header &GeometryMesh::GetHeader() const {
  return header_;
}

}  // namespace sparkium
