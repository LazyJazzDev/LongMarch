#pragma once
#include "xing_huo/core/core_util.h"

namespace XH {

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

}  // namespace XH
