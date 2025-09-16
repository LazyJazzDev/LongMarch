#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanShader : public Shader {
 public:
  VulkanShader(VulkanCore *core, const CompiledShaderBlob &shader_blob);
  ~VulkanShader() override = default;

  vulkan::ShaderModule *ShaderModule() const {
    return shader_module_.get();
  }

  const std::string &EntryPoint() const override {
    return shader_module_->EntryPoint();
  }

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::ShaderModule> shader_module_;
};

}  // namespace grassland::graphics::backend
