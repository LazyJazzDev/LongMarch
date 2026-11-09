#pragma once
#include "grassland/rhi/backend/backend_shader.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Shader final : public Shader {
 public:
  ShaderStage GetStage() const override {
    return ShaderStageValue();
  }

  const std::string &GetEntry() const override {
    return EntryPoint();
  }

  const std::string &GetPreprocessedCode() const override {
    return PreprocessedSource();
  }

  static D3D12Shader *Create(const ShaderDesc &desc) {
    auto shader = std::unique_ptr<D3D12Shader>(new D3D12Shader(desc));
    if (!shader->InitializeSource(desc, BACKEND_API_D3D12))
      return nullptr;
    return shader.release();
  }

 private:
  explicit D3D12Shader(const ShaderDesc &desc) : Shader(desc) {
  }
};
}  // namespace grassland::rhi::backend::d3d12
