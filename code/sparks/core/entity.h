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
  virtual bool ExpiredBuffer();
  virtual bool ExpiredImage();
  virtual bool ExpiredHitGroup();
  virtual bool ExpiredCallableShader();

  operator bool() const {
    return core_ != nullptr;
  }

 protected:
  Core *core_;
};

}  // namespace sparks
