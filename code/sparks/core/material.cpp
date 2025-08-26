#include "sparks/core/material.h"

#include "sparks/core/core.h"

namespace sparks {

Material::Material(Core *core) : core_(core) {
}

const CodeLines &Material::PowerSamplerImpl() const {
  return {};
}

}  // namespace sparks
