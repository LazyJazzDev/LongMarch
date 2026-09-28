#include "grassland/graphics/backend/vulkan/vulkan_shader.h"

namespace grassland::graphics::backend {

VulkanShader::VulkanShader(VulkanCore *core, const CompiledShaderBlob &shader_blob) : core_(core) {
  if (shader_blob.data.size() % sizeof(uint32_t))
    throw std::invalid_argument("Invalid SPIR-V byte count");
  std::vector<uint32_t> words(shader_blob.data.size() / sizeof(uint32_t));
  std::memcpy(words.data(), shader_blob.data.data(), shader_blob.data.size());
  VkShaderModuleCreateInfo create_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  create_info.codeSize = words.size() * sizeof(uint32_t);
  create_info.pCode = words.data();
  vulkan::ThrowIfFailed(vkCreateShaderModule(core_->Handle(), &create_info, nullptr, &shader_module_),
                        "Failed to create Vulkan shader module");
  entry_point_ = shader_blob.entry_point;
}

VulkanShader::~VulkanShader() {
  vkDestroyShaderModule(core_->Handle(), shader_module_, nullptr);
}

}  // namespace grassland::graphics::backend
