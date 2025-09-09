#include "xing_huo/core/material.h"

#include "xing_huo/core/core.h"

namespace XH {

Material::Material(Core *core) : core_(core) {
}

void Material::Update(Scene *scene) {
}

const CodeLines &Material::EvaluatorImpl() const {
  return {};
}

}  // namespace XH
