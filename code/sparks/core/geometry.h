#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Geometry {
 public:
  Geometry(Core *core);
  virtual ~Geometry() = default;

  virtual graphics::Buffer *Buffer() = 0;
  virtual graphics::AccelerationStructure *BLAS() = 0;
  virtual graphics::HitGroup HitGroup() = 0;
  virtual int PrimitiveCount() = 0;
  virtual const CodeLines &SamplerImplementation() const;

 protected:
  Core *core_;
};

}  // namespace sparks
