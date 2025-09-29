#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

class Geometry : public Object {
 public:
  Geometry(Core *core);

  Core *GetCore() const;

  virtual ~Geometry() = default;

  virtual int PrimitiveCount() = 0;

 protected:
  Core *core_;
};

}  // namespace sparkium
