#include "sparkium/backend/cuda/path_tracing/geometry/geometry_mesh.h"

#include "sparkium/backend/cuda/path_tracing/core/core.h"

namespace sparkium::cuda_tracing {

GeometryMesh::GeometryMesh(sparkium::GeometryMesh &geometry)
    : geometry_(geometry),
      Geometry(DedicatedCast(geometry.GetCore())) {
  auto &vfs = core_->GetShadersVFS();
  sampler_implementation_ = CodeLines(vfs, "geometry/mesh/geometry_sampler.hlsli");
}

graphics::Buffer *GeometryMesh::Buffer() {
  return geometry_.GetBuffer();
}

const Mesh<float> *GeometryMesh::HostMesh() const {
  return geometry_.HostMesh();
}

int GeometryMesh::PrimitiveCount() {
  return geometry_.PrimitiveCount();
}

const CodeLines &GeometryMesh::SamplerImpl() const {
  return sampler_implementation_;
}

graphics::AccelerationStructure *GeometryMesh::BLAS() {
  if (!blas_) {
    auto header = geometry_.GetHeader();

    core_->BackendDevice()->CreateBottomLevelAccelerationStructure(
        geometry_.GetBuffer()->Range(header.position_offset), geometry_.GetBuffer()->Range(header.index_offset),
        header.num_vertices, header.position_stride, header.num_indices / 3, graphics::RAYTRACING_GEOMETRY_FLAG_NONE,
        &blas_);
  }
  return blas_.get();
}

}  // namespace sparkium::cuda_tracing
