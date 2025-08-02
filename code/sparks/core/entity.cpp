#include "sparks/core/entity.h"

#include "scene.h"

namespace sparks {

bool Entity::ExpiredBuffer() {
  return false;
}

bool Entity::ExpiredImage() {
  return false;
}

bool Entity::ExpiredHitGroup() {
  return false;
}

bool Entity::ExpiredCallableShader() {
  return false;
}

}  // namespace sparks
