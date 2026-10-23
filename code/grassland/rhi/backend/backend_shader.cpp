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
  shader->compiled_desc_ = desc;
  shader->compiled_desc_.file_system = nullptr;
  if (!PreprocessShader(desc, api, shader->compiled_desc_.source)) {
    delete shader;
    return nullptr;
  }
  return shader;
}

const ShaderDesc &Shader::CompileDesc() const {
  return compiled_desc_;
}

const ShaderDesc &ShaderCompileDesc(const IShader *shader) {
  return static_cast<const Shader *>(shader)->CompileDesc();
}
}  // namespace grassland::rhi::backend
