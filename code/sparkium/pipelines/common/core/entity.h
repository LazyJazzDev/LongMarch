#pragma once
#include "sparkium/pipelines/common/core/core_util.h"

namespace sparkium::render_shared {
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
}  // namespace sparkium::render_shared
