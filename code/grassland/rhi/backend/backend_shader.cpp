#include "grassland/rhi/backend/backend_shader.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/backend_shader_impl.h"

namespace grassland::rhi::backend {
ShaderImpl::ShaderImpl(const ShaderDesc &desc) : stage_(desc.stage), entry_(desc.entry_point) {
}

ShaderStage ShaderImpl::GetStage() const {
  return stage_;
}

const std::string &ShaderImpl::GetEntry() const {
  return entry_;
}

rhi::IShader *ShaderImpl::Create(const ShaderDesc &desc, BackendAPI api) {
  auto shader = new ShaderImpl(desc);
  if (!PreprocessShader(desc, api, shader->preprocessed_code_)) {
    delete shader;
    return nullptr;
  }
  return shader;
}

const std::string &ShaderImpl::GetPreprocessedCode() const {
  return preprocessed_code_;
}

rhi::IShader *CreateShader(const ShaderDesc &desc, BackendAPI api) {
  return ShaderImpl::Create(desc, api);
}
}  // namespace grassland::rhi::backend
