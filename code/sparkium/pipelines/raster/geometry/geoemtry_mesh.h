#pragma once
#include "sparkium/pipelines/raster/core/geometry.h"

namespace sparkium::raster {

class GeometryMesh : public Geometry {
 public:
  GeometryMesh(sparkium::GeometryMesh &geometry);
  graphics::Shader *VertexShader() override;
  void SetupProgram(graphics::Program *program) override;
  void DispatchDrawCalls(graphics::CommandContext *cmd_ctx) override;

 private:
  sparkium::GeometryMesh &geometry_;
  std::unique_ptr<graphics::Shader> vertex_shader_;
};

}  // namespace sparkium::raster
