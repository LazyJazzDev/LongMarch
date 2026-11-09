#pragma once
#include "grassland/rhi/backend/backend_shader.h"

namespace grassland::rhi::backend::metal {
class MetalShader final : public Shader {
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

  static MetalShader *Create(const ShaderDesc &desc) {
    auto shader = std::unique_ptr<MetalShader>(new MetalShader(desc));
    if (!shader->InitializeSource(desc, BACKEND_API_METAL))
      return nullptr;
    return shader.release();
  }

 private:
  explicit MetalShader(const ShaderDesc &desc) : Shader(desc) {
  }
};
}  // namespace grassland::rhi::backend::metal
