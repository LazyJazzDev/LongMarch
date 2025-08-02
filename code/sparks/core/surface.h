#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Surface {
 public:
  Surface(Core *core);

  virtual graphics::Buffer *Buffer() = 0;
  virtual graphics::Shader *CallableShader() = 0;
  virtual const CodeLines &SamplerImpl() const = 0;
  virtual const CodeLines &EvaluatorImpl() const;

 protected:
  Core *core_;
};

}  // namespace sparks
