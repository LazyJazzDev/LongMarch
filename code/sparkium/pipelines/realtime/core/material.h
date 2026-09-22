#pragma once
#include "sparkium/pipelines/realtime/core/core_util.h"

namespace sparkium::realtime {

class Material : public Object {
 public:
  Material(Core *core);
  virtual ~Material() = default;

  virtual void Update(Scene *scene);
  virtual graphics::Buffer *Buffer() = 0;
  virtual const CodeLines &SamplerImpl() const = 0;
  virtual const CodeLines &EvaluatorImpl() const;

  // Graph parameters can be dispatched separately from the shared BSDF sampler.
  virtual const CodeLines *GraphImpl() const {
    return nullptr;
  }

 protected:
  Core *core_;
};

}  // namespace sparkium::realtime
