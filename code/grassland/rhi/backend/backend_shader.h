#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi::backend {
class Shader : public Object<IShader> {
 protected:
  bool InitializeSource(const ShaderDesc &desc, BackendAPI api);
  ShaderStage ShaderStageValue() const;
  const std::string &EntryPoint() const;
  const std::string &PreprocessedSource() const;

  explicit Shader(const ShaderDesc &desc);

 private:
  ShaderStage stage_;
  std::string entry_;
  std::string preprocessed_code_;
};
}  // namespace grassland::rhi::backend
