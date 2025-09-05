#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Material {
 public:
  Material(Core *core);
  virtual ~Material() = default;

  virtual void Update(Scene *scene);
  virtual graphics::Buffer *Buffer() = 0;
  virtual const CodeLines &SamplerImpl() const = 0;
  virtual const CodeLines &EvaluatorImpl() const;

 protected:
  Core *core_;
};

}  // namespace sparks
