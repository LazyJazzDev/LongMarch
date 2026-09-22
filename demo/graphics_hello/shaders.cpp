#include "module.h"

namespace {
#include "built_in_shaders.inl"
}

namespace graphics_hello {
std::string LoadShader(const std::string &path) {
  const auto &shader = shader_list.at(path);
  return std::string(shader.first, shader.second);
}
}  // namespace graphics_hello
