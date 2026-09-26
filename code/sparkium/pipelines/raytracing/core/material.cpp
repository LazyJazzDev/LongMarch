#include "sparkium/pipelines/raytracing/core/material.h"

#include <cstring>

namespace sparkium::raytracing {

Material::Material(Core *core) : core_(core) {
}

void Material::UploadMaterialData(graphics::Buffer *buffer, const void *data, size_t size, size_t offset) {
  if (!size)
    return;
  auto &uploaded = uploaded_data_[offset];
  if (uploaded.size() == size && std::memcmp(uploaded.data(), data, size) == 0)
    return;
  buffer->UploadData(data, size, offset);
  uploaded.resize(size);
  std::memcpy(uploaded.data(), data, size);
}

void Material::Update(Scene *scene) {
}

const CodeLines &Material::EvaluatorImpl() const {
  static CodeLines empty_code_lines;
  return empty_code_lines;
}

}  // namespace sparkium::raytracing
