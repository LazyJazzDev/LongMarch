#include "sparks/core/surface.h"

#include "sparks/core/core.h"

namespace sparks {

Surface::Surface(Core *core) : core_(core) {
}

const CodeLines &Surface::SamplerImplementation() const {
  return {};
}

}  // namespace sparks
