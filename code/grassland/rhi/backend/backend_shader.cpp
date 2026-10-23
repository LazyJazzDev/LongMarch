#include "grassland/rhi/backend/backend_shader.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"

namespace grassland::rhi::backend {
Shader::Shader(const ShaderDesc &desc) : desc_(desc) {
}

const ShaderDesc &Shader::GetDesc() const {
  return desc_;
}

Shader *Shader::Create(const ShaderDesc &desc, BackendAPI api) {
  auto shader = new Shader(desc);
  if (!PreprocessShader(desc, api, shader->preprocessed_shader_code_)) {
    delete shader;
    return nullptr;
  }
  return shader;
}

const std::string &Shader::GetPreprocessedShaderCode() const {
  return preprocessed_shader_code_;
}

}  // namespace grassland::rhi::backend
