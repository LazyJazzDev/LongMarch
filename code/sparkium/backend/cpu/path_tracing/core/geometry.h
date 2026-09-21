#pragma once
#include "sparkium/backend/cpu/path_tracing/core/core_util.h"

namespace sparkium::cpu_tracing {

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

}  // namespace sparkium::cpu_tracing
