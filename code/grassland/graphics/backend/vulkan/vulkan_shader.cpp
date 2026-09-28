#include "grassland/graphics/backend/vulkan/vulkan_shader.h"

namespace grassland::graphics::backend {

VulkanShader::VulkanShader(VulkanCore *core, const CompiledShaderBlob &shader_blob) : core_(core) {
  VkShaderModuleCreateInfo create_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  create_info.codeSize = shader_blob.data.size();
  create_info.pCode = reinterpret_cast<const uint32_t *>(shader_blob.data.data());
  vulkan::ThrowIfFailed(vkCreateShaderModule(core_->Handle(), &create_info, nullptr, &shader_module_),
                        "Failed to create Vulkan shader module");
  entry_point_ = shader_blob.entry_point;
}

VulkanShader::~VulkanShader() {
  vkDestroyShaderModule(core_->Handle(), shader_module_, nullptr);
}

}  // namespace grassland::graphics::backend
