#include "xing_huo/geometry/geometry_mesh.h"

#include "xing_huo/core/core.h"

namespace XH {

GeometryMesh::GeometryMesh(Core *core, const Mesh<float> &mesh) : Geometry(core) {
  std::vector<uint8_t> data;
  auto write_data = [&](const void *data_ptr, size_t size) {
    data.insert(data.end(), static_cast<const uint8_t *>(data_ptr), static_cast<const uint8_t *>(data_ptr) + size);
  };
  GeometryMeshHeader header{};
  write_data(&header, sizeof(header));

  Mesh<float> mesh_copy;
  const Mesh<float> *mesh_ptr = &mesh;
  if (mesh.Normals() && mesh.TexCoords()) {
    mesh_copy = mesh;
    mesh_copy.GenerateTangents();
    mesh_ptr = &mesh_copy;
  }

  header.num_indices = mesh_ptr->NumIndices();
  header.num_vertices = mesh_ptr->NumVertices();

  header.index_offset = data.size();
  write_data(mesh_ptr->Indices(), mesh_ptr->NumIndices() * sizeof(uint32_t));
  header.position_offset = data.size();
  header.position_stride = sizeof(float) * 3;
  write_data(mesh_ptr->Positions(), mesh_ptr->NumVertices() * sizeof(float) * 3);

  if (mesh_ptr->Normals()) {
    header.normal_offset = data.size();
    header.normal_stride = sizeof(float) * 3;
    write_data(mesh_ptr->Normals(), mesh_ptr->NumVertices() * sizeof(float) * 3);
  }

  if (mesh_ptr->TexCoords()) {
    header.tex_coord_offset = data.size();
    header.tex_coord_stride = sizeof(float) * 2;
    write_data(mesh_ptr->TexCoords(), mesh_ptr->NumVertices() * sizeof(float) * 2);
  }

  if (mesh_ptr->Signals()) {
    header.signal_offset = data.size();
    header.signal_stride = sizeof(float);
    write_data(mesh_ptr->Signals(), mesh_ptr->NumVertices() * sizeof(float));
  }

  std::memcpy(data.data(), &header, sizeof(header));

  core_->GraphicsCore()->CreateBuffer(data.size(), graphics::BUFFER_TYPE_STATIC, &geometry_buffer_);
  geometry_buffer_->UploadData(data.data(), data.size());
  core_->GraphicsCore()->CreateBottomLevelAccelerationStructure(
      geometry_buffer_->Range(header.position_offset), geometry_buffer_->Range(header.index_offset),
      header.num_vertices, header.position_stride, header.num_indices / 3, graphics::RAYTRACING_GEOMETRY_FLAG_NONE,
      &blas_);

  auto &vfs = core_->GetShadersVFS();
  primitive_count_ = header.num_indices / 3;
  sampler_implementation_ = CodeLines(vfs, "geometry/mesh/geometry_sampler.hlsli");
  closest_hit_shader_implementation_ = CodeLines(vfs, "geometry/mesh/hit_group.hlsl");
}

graphics::Buffer *GeometryMesh::Buffer() {
  return geometry_buffer_.get();
}

const CodeLines &GeometryMesh::ClosestHitShaderImpl() const {
  return closest_hit_shader_implementation_;
}

int GeometryMesh::PrimitiveCount() {
  return primitive_count_;
}

const CodeLines &GeometryMesh::SamplerImpl() const {
  return sampler_implementation_;
}

graphics::AccelerationStructure *GeometryMesh::BLAS() {
  return blas_.get();
}

}  // namespace XH
