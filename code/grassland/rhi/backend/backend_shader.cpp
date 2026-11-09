#include "grassland/rhi/backend/backend_shader.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"

namespace grassland::rhi::backend {
Shader::Shader(const ShaderDesc &desc) : stage_(desc.stage), entry_(desc.entry_point) {
}

ShaderStage Shader::ShaderStageValue() const {
  return stage_;
}

const std::string &Shader::EntryPoint() const {
  return entry_;
}

const std::string &Shader::PreprocessedSource() const {
  return preprocessed_code_;
}

bool Shader::InitializeSource(const ShaderDesc &desc, BackendAPI api) {
  if (desc.filename.empty() || desc.entry_point.empty() || desc.stage < SHADER_STAGE_VERTEX ||
      desc.stage > SHADER_STAGE_CALLABLE)
    return false;
  return PreprocessShader(desc, api, preprocessed_code_);
}

}  // namespace grassland::rhi::backend
