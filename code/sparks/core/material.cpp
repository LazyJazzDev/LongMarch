#include "sparks/core/material.h"

#include "sparks/core/core.h"

namespace sparks {

Material::Material(Core *core) : core_(core) {
}

void Material::Update(Scene *scene) {
}

const CodeLines &Material::EvaluatorImpl() const {
  return {};
}

}  // namespace sparks
