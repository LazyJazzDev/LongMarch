#include "practium/core/entity.h"

#include "scene.h"

namespace practium {

Entity::Entity(Scene *scene) : scene_(scene) {
  scene_->RegisterEntity(this);
}

Scene *Entity::GetScene() const {
  return scene_;
}

}  // namespace practium
