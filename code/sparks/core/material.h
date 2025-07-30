#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Material {
 public:
  Material(Core *core);

  virtual graphics::Buffer *Buffer() = 0;
  virtual graphics::Shader *CallableShader() = 0;

 protected:
  Core *core_;
};

}  // namespace sparks
