#include "grassland/rhi/backend/backend_shader.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"

namespace grassland::rhi::backend {
Shader::Shader(const ShaderDesc &desc) : stage_(desc.stage), entry_(desc.entry_point) {
}

ShaderStage Shader::GetStage() const {
  return stage_;
}

const std::string &Shader::GetEntry() const {
  return entry_;
}

Shader *Shader::Create(const ShaderDesc &desc, BackendAPI api) {
  auto shader = new Shader(desc);
  if (!PreprocessShader(desc, api, shader->preprocessed_code_)) {
    delete shader;
    return nullptr;
  }
  return shader;
}

const std::string &Shader::GetPreprocessedCode() const {
  return preprocessed_code_;
}

}  // namespace grassland::rhi::backend
