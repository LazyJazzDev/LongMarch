#include "sparks/core/light.h"

namespace sparks {

Light::Light(Core *core) : core_(core) {
}

graphics::Buffer *Light::GeometryData() {
  return nullptr;
}

}  // namespace sparks
