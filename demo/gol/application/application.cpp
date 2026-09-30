#include "application.h"

namespace {
#include "built_in_shaders.inl"
}

std::string DemoSurfaceShader() {
  return GetShaderCode("shaders/super.slang");
}
