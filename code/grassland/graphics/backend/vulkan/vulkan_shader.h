#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanShader : public Shader {
 public:
  VulkanShader(VulkanCore *core, const CompiledShaderBlob &shader_blob);
  ~VulkanShader() override;

  VkShaderModule ModuleHandle() const {
    return shader_module_;
  }

  std::string EntryPoint() const override {
    return entry_point_;
  }

  const std::string &EntryPointRef() const {
    return entry_point_;
  }

 private:
  VulkanCore *core_;
  VkShaderModule shader_module_{VK_NULL_HANDLE};
  std::string entry_point_;
};

}  // namespace grassland::graphics::backend
