#include "sparks/core/geometry.h"

#include "sparks/core/core.h"

namespace sparks {
Geometry::Geometry(Core *core) : core_(core) {
}

const CodeLines &Geometry::SamplerImplementation() const {
  return {};
}
}  // namespace sparks
