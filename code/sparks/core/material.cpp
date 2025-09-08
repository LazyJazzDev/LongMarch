#include "sparks/core/material.h"

#include "sparks/core/core.h"

namespace XH {

Material::Material(Core *core) : core_(core) {
}

void Material::Update(Scene *scene) {
}

const CodeLines &Material::EvaluatorImpl() const {
  return {};
}

}  // namespace XH
