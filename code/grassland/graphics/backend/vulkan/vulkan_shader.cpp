#include "grassland/graphics/backend/vulkan/vulkan_shader.h"

namespace grassland::graphics::backend {

VulkanShader::VulkanShader(VulkanCore *core, const CompiledShaderBlob &shader_blob) : core_(core) {
  core_->Device()->CreateShaderModule(shader_blob, &shader_module_);
}

}  // namespace grassland::graphics::backend
