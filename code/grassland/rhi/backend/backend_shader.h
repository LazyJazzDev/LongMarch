#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi::backend {
class Shader final : public Object<IShader> {
 public:
  static Shader *Create(const ShaderDesc &desc, BackendAPI api);
  ShaderStage GetStage() const override;
  const std::string &GetEntry() const override;
  const std::string &GetPreprocessedCode() const override;

 private:
  explicit Shader(const ShaderDesc &desc);
  ShaderStage stage_;
  std::string entry_;
  std::string preprocessed_code_;
};
}  // namespace grassland::rhi::backend
