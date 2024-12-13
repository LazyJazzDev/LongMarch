#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {
class VulkanShader : public Shader {
 public:
  VulkanShader(VulkanCore *core, const void *data, size_t size);

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::ShaderModule> shader_module_;
};
}  // namespace grassland::graphics::backend
