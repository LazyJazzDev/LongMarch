#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Entity {
 public:
  Entity(Scene *scene);

  virtual ~Entity() = default;

  Scene *GetScene() const;

 protected:
  Scene *scene_;
};

}  // namespace practium
