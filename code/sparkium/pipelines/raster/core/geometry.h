#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Geometry : public Object {
 public:
  explicit Geometry(Core *core) : core_(core) {
  }
  virtual ~Geometry() = default;
  virtual graphics::Buffer *VertexBuffer() = 0;
  virtual graphics::Buffer *IndexBuffer() = 0;
  virtual graphics::Shader *VertexShader() = 0;

 protected:
  Core *core_{};
};

}  // namespace sparkium::raster
