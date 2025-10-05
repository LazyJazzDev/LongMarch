#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Material {
 public:
  Material(Core *core);
  virtual ~Material() = default;

 protected:
  Core *core_;
};

}  // namespace practium
