#include "sparkium/core/material.h"

#include "sparkium/core/core.h"

namespace sparkium {

Material::Material(Core *core) : core_(core) {
}

Core *Material::GetCore() const {
  return core_;
}

}  // namespace sparkium
