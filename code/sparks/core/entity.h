#pragma once
#include "sparks/core/core_util.h"
#include "sparks/core/light.h"

namespace XH {

class Entity {
 public:
  Entity(Core *core) : core_(core) {
  }
  virtual ~Entity() = default;
  virtual void Update(Scene *scene) = 0;

  operator bool() const {
    return core_ != nullptr;
  }

 protected:
  Core *core_;
};

}  // namespace XH
