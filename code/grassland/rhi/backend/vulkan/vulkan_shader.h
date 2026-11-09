#pragma once
#include "grassland/rhi/backend/backend_shader.h"

namespace grassland::rhi::backend::vulkan {
class VulkanShader final : public Shader {
 public:
  void AddRef() noexcept override {
    RetainReference();
  }

  void Release() noexcept override {
    if (ReleaseReference())
      delete this;
  }

  ShaderStage GetStage() const override {
    return ShaderStageValue();
  }

  const std::string &GetEntry() const override {
    return EntryPoint();
  }

  const std::string &GetPreprocessedCode() const override {
    return PreprocessedSource();
  }

  static VulkanShader *Create(const ShaderDesc &desc) {
    auto shader = std::unique_ptr<VulkanShader>(new VulkanShader(desc));
    if (!shader->InitializeSource(desc, BACKEND_API_VULKAN))
      return nullptr;
    return shader.release();
  }

 private:
  explicit VulkanShader(const ShaderDesc &desc) : Shader(desc) {
  }
};
}  // namespace grassland::rhi::backend::vulkan
