#pragma once
#include "sparkium/pipelines/realtime/core/core_util.h"

namespace sparkium::realtime {

class Geometry : public Object {
 public:
  Geometry(Core *core);
  virtual ~Geometry() = default;

  virtual graphics::Buffer *Buffer() = 0;
  virtual int PrimitiveCount() = 0;
  virtual const CodeLines &SamplerImpl() const = 0;

 protected:
  Core *core_;
};

}  // namespace sparkium::realtime
