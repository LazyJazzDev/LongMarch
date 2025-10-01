#pragma once
#include "sparkium/pipelines/raster/core/geometry.h"

namespace sparkium::raster {

class GeometryMesh : public Geometry {
 public:
  GeometryMesh(sparkium::GeometryMesh &geometry);
  graphics::Buffer *VertexBuffer() override;
  graphics::Buffer *IndexBuffer() override;
  graphics::Shader *VertexShader() override;

 private:
  sparkium::GeometryMesh &geometry_;
};

}  // namespace sparkium::raster
