#include "grassland/rhi/rhix/rhix_shader.h"

namespace grassland::rhix {
Shader::Shader() noexcept = default;

Shader::Shader(rhi::IShader *object) noexcept : ObjectContainer(object) {
}

rhi::IShader *Shader::Get() const noexcept {
  return static_cast<rhi::IShader *>(GetObject());
}

const rhi::ShaderDesc &Shader::GetDesc() const {
  return Get()->GetDesc();
}
}  // namespace grassland::rhix
