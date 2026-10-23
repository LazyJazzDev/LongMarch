#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi::backend {
class Shader final : public Object<IShader> {
 public:
  static Shader *Create(const ShaderDesc &desc, BackendAPI api);
  const std::string &GetPreprocessedShaderCode() const;
  const ShaderDesc &GetDesc() const override;

 private:
  explicit Shader(const ShaderDesc &desc);
  ShaderDesc desc_;
  std::string preprocessed_shader_code_;
};
}  // namespace grassland::rhi::backend
