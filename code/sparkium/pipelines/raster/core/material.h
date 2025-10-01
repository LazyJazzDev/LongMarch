#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Material : public Object {
 public:
  explicit Material(Core *core) : core_(core) {
  }
  virtual ~Material() = default;
  virtual graphics::Shader *PixelShader() = 0;

 protected:
  Core *core_{};
};

}  // namespace sparkium::raster
