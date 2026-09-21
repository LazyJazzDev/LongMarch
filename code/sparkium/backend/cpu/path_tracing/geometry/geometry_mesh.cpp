#include "sparkium/backend/cpu/path_tracing/geometry/geometry_mesh.h"

#include "sparkium/backend/cpu/path_tracing/core/core.h"

namespace sparkium::cpu_tracing {

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

}  // namespace sparkium::cpu_tracing
