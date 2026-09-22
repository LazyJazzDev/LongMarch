#include "sparkium/pipelines/realtime/geometry/geometry_mesh.h"

#include "sparkium/pipelines/realtime/core/core.h"

namespace sparkium::realtime {

GeometryMesh::GeometryMesh(sparkium::GeometryMesh &geometry)
    : geometry_(geometry),
      Geometry(DedicatedCast(geometry.GetCore())) {
  auto &vfs = core_->GetShadersVFS();
  sampler_implementation_ = CodeLines(vfs, "geometry/mesh/geometry_sampler.hlsli");
}

graphics::Buffer *GeometryMesh::Buffer() {
  return geometry_.GetBuffer();
}

int GeometryMesh::PrimitiveCount() {
  return geometry_.PrimitiveCount();
}

const CodeLines &GeometryMesh::SamplerImpl() const {
  return sampler_implementation_;
}

}  // namespace sparkium::realtime
