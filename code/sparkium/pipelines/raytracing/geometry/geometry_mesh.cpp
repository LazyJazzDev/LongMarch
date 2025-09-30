#include "sparkium/pipelines/raytracing/geometry/geometry_mesh.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

GeometryMesh::GeometryMesh(sparkium::GeometryMesh &geometry)
    : geometry_(geometry), Geometry(DedicatedCast(geometry.GetCore())) {
  auto header = geometry_.GetHeader();

  core_->GraphicsCore()->CreateBottomLevelAccelerationStructure(
      geometry_.GetBuffer()->Range(header.position_offset), geometry_.GetBuffer()->Range(header.index_offset),
      header.num_vertices, header.position_stride, header.num_indices / 3, graphics::RAYTRACING_GEOMETRY_FLAG_NONE,
      &blas_);

  auto &vfs = core_->GetShadersVFS();
  sampler_implementation_ = CodeLines(vfs, "geometry/mesh/geometry_sampler.hlsli");
  closest_hit_shader_implementation_ = CodeLines(vfs, "geometry/mesh/hit_group.hlsl");
}

graphics::Buffer *GeometryMesh::Buffer() {
  return geometry_.GetBuffer();
}

const CodeLines &GeometryMesh::ClosestHitShaderImpl() const {
  return closest_hit_shader_implementation_;
}

int GeometryMesh::PrimitiveCount() {
  return geometry_.PrimitiveCount();
}

const CodeLines &GeometryMesh::SamplerImpl() const {
  return sampler_implementation_;
}

graphics::AccelerationStructure *GeometryMesh::BLAS() {
  return blas_.get();
}

}  // namespace sparkium::raytracing
