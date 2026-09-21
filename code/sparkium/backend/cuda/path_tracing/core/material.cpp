#include "sparkium/backend/cuda/path_tracing/core/material.h"

namespace sparkium::cuda_tracing {

Material::Material(Core *core) : core_(core) {
}

void Material::Update(Scene *scene) {
}

const CodeLines &Material::EvaluatorImpl() const {
  static CodeLines empty_code_lines;
  return empty_code_lines;
}

}  // namespace sparkium::cuda_tracing
