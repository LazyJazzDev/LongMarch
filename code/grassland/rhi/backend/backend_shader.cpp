#include "grassland/rhi/backend/backend_shader.h"

namespace grassland::rhi::backend {
Shader::Shader(const ShaderDesc &desc) : desc_(desc) {
}

const ShaderDesc &Shader::GetDesc() const {
  return desc_;
}
}  // namespace grassland::rhi::backend
