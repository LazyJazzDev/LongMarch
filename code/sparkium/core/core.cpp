#include "sparkium/core/core.h"

#include "sparkium/core/camera.h"
#include "sparkium/core/entity.h"
#include "sparkium/core/film.h"
#include "sparkium/core/geometry.h"
#include "sparkium/core/material.h"
#include "sparkium/core/scene.h"

namespace sparkium {
Core::Core(graphics::Core *core) : core_(core) {
}

graphics::Core *Core::GraphicsCore() const {
  return core_;
}

}  // namespace sparkium
