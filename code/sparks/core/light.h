#pragma once
#include "sparks/core/core_util.h"

namespace sparks {
class Light {
 public:
  Light(Core *core);
  virtual ~Light() = default;
  virtual graphics::Shader *SamplerShader() = 0;
  virtual graphics::Buffer *SamplerData() = 0;

 protected:
  Core *core_;
};
}  // namespace sparks
