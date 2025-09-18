#include "sparkium/core/material.h"

#include "sparkium/core/core.h"

namespace sparkium {

Material::Material(Core *core) : core_(core) {
}

void Material::Update(Scene *scene) {
}

const CodeLines &Material::EvaluatorImpl() const {
  return {};
}

}  // namespace sparkium
