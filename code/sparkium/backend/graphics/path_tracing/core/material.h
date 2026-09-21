#pragma once
#include "sparkium/backend/graphics/path_tracing/core/core_util.h"

namespace sparkium::raytracing {

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

}  // namespace sparkium::raytracing
