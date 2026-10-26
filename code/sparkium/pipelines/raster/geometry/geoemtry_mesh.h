#pragma once
#include "sparkium/pipelines/raster/core/geometry.h"

namespace sparkium::raster {

class GeometryMesh : public Geometry {
 public:
  GeometryMesh(sparkium::GeometryMesh &geometry);
  rhi::IShader *VertexShader() override;
  void SetupProgram(rhi::ProgramDesc &program) override;
  void DispatchDrawCalls(rhi::ICommandContext *cmd_ctx) override;
  glm::vec4 CentricArea(const glm::mat4x3 &affine) override;

 private:
  sparkium::GeometryMesh &geometry_;
  rhi::ObjPtr<rhi::IShader> vertex_shader_;
};

}  // namespace sparkium::raster
