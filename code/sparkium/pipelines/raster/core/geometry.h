#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Geometry : public Object {
 public:
  explicit Geometry(Core *core) : core_(core) {
  }

  virtual ~Geometry() = default;
  virtual rhi::IShader *VertexShader() = 0;
  virtual void SetupProgram(rhi::ProgramDesc &program);
  virtual void DispatchDrawCalls(rhi::ICommandContext *cmd_ctx);
  virtual glm::vec4 CentricArea(const glm::mat4x3 &affine) = 0;

 protected:
  Core *core_{};
};

}  // namespace sparkium::raster
