#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Material {
 public:
  Material(Core *core);
  virtual ~Material() = default;

  virtual graphics::Buffer *Buffer() = 0;
  virtual const CodeLines &SamplerImpl() const = 0;
  virtual const CodeLines &EvaluatorImpl() const = 0;
  virtual const CodeLines &PowerSamplerImpl() const;

 protected:
  Core *core_;
};

}  // namespace sparks
