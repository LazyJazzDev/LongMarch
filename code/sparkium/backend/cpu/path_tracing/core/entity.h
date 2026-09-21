#pragma once
#include "sparkium/backend/cpu/path_tracing/core/core_util.h"

namespace sparkium::cpu_tracing {
class Entity : public Object {
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
}  // namespace sparkium::cpu_tracing
