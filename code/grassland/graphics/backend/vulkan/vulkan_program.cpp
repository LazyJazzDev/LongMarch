#include "grassland/graphics/backend/vulkan/vulkan_program.h"

namespace grassland::graphics::backend {
VulkanShader::VulkanShader(VulkanCore *core, const void *data, size_t size)
    : core_(core) {
  core_->Device()->CreateShaderModule(data, size, &shader_module_);
}
}  // namespace grassland::graphics::backend
