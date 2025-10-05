#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Entity {
 public:
  Entity(Scene *scene);

  virtual ~Entity() = default;

  Scene *GetScene() const;

  virtual void SyncRenderState() const = 0;

 protected:
  Scene *scene_;
};

}  // namespace practium
