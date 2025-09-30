#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

class Material : public Object {
 public:
  Material(Core *core);

  Core *GetCore() const;

  virtual ~Material() = default;

 protected:
  Core *core_;
};

}  // namespace sparkium
