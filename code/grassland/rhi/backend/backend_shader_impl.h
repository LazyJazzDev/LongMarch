#pragma once
#include "grassland/rhi/backend/backend_shader.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
class ShaderImpl final : public Object<rhi::IShader> {
 public:
  static rhi::IShader *Create(const ShaderDesc &desc, BackendAPI api);
  ShaderStage GetStage() const override;
  const std::string &GetEntry() const override;
  const std::string &GetPreprocessedCode() const override;

 private:
  explicit ShaderImpl(const ShaderDesc &desc);
  ShaderStage stage_;
  std::string entry_;
  std::string preprocessed_code_;

 public:
  ~ShaderImpl() override = default;
};
}  // namespace grassland::rhi::backend
