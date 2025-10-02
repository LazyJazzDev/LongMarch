#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Scene;

class Entity : public Object {
 public:
  explicit Entity(Core *core) : core_(core) {
  }
  virtual ~Entity() = default;
  virtual void Update(Scene *scene) = 0;
  operator bool() const {
    return core_ != nullptr;
  }

 protected:
  Core *core_{};
};

}  // namespace sparkium::raster
