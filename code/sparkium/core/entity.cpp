#include "sparkium/core/entity.h"

#include "sparkium/core/scene.h"

namespace sparkium {

Entity::Entity(Core *core) : core_(core) {
}

Core *Entity::GetCore() const {
  return core_;
}

}  // namespace sparkium
