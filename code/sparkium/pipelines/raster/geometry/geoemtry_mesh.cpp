#include "sparkium/pipelines/raster/geometry/geoemtry_mesh.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

GeometryMesh::GeometryMesh(sparkium::GeometryMesh &geometry)
    : geometry_(geometry), Geometry(DedicatedCast(geometry.GetCore())) {
}

graphics::Buffer *GeometryMesh::VertexBuffer() {
  return nullptr;
}

graphics::Buffer *GeometryMesh::IndexBuffer() {
  return nullptr;
}

graphics::Shader *GeometryMesh::VertexShader() {
  return nullptr;
}

}  // namespace sparkium::raster
