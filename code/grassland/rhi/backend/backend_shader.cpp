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

IBackendShader *ShaderImpl::Create(const ShaderDesc &desc, BackendAPI api) {
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

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendShader, "74166d66-39c8-4b6f-b9bc-1721e275a3c2")

namespace grassland::rhi::backend {
IBackendShader *IBackendShader::Create(const ShaderDesc &desc, BackendAPI api) {
  return ShaderImpl::Create(desc, api);
}
}  // namespace grassland::rhi::backend
