#include "sparkium/core/geometry.h"

#include "sparkium/core/core.h"

namespace sparkium {

Geometry::Geometry(Core *core) : core_(core) {
}

Core *Geometry::GetCore() const {
  return core_;
}

}  // namespace sparkium
