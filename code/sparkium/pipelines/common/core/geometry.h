#pragma once
#include "sparkium/pipelines/common/core/core_util.h"

namespace sparkium::render_shared {

class Geometry : public Object {
 public:
  Geometry(Core *core);
  virtual ~Geometry() = default;

  virtual graphics::Buffer *Buffer() = 0;
  virtual graphics::AccelerationStructure *BLAS() = 0;
  virtual const CodeLines &ClosestHitShaderImpl() const = 0;
  virtual int PrimitiveCount() = 0;
  virtual const CodeLines &SamplerImpl() const = 0;

 protected:
  Core *core_;
};

}  // namespace sparkium::render_shared
