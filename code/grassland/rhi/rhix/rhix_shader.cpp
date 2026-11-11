#include "grassland/rhi/rhix/rhix_shader.h"

namespace grassland::rhix {
Shader::Shader() noexcept = default;

Shader::Shader(rhi::IShader *object) noexcept : ObjectContainer(object) {
}

rhi::IShader *Shader::Get() const noexcept {
  return rhi::QueryInterfacePtr<rhi::IShader>(GetObject()).Get();
}

rhi::ShaderStage Shader::GetStage() const {
  return Get()->GetStage();
}

const std::string &Shader::GetEntry() const {
  return Get()->GetEntry();
}

const std::string &Shader::GetPreprocessedCode() const {
  return Get()->GetPreprocessedCode();
}
}  // namespace grassland::rhix
