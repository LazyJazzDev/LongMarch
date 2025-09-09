#pragma once
#include "xing_huo/core/core_util.h"
#include "xing_huo/core/light.h"

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
