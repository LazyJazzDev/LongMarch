#include "practium/core/entity.h"

namespace practium {

Entity::Entity(Scene *scene) : scene_(scene) {
}

Scene *Entity::GetScene() const {
  return scene_;
}

}  // namespace practium
