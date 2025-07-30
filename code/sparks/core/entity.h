#pragma once
#include "sparks/core/core_util.h"
#include "sparks/core/light.h"

namespace sparks {

class Entity {
 public:
  Entity(Core *core) : core_(core) {
  }
  virtual ~Entity() = default;
  virtual void Update(Scene *scene) = 0;

 protected:
  Core *core_;
};

}  // namespace sparks
