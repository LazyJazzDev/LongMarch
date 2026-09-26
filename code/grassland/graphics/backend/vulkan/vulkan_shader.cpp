#include "grassland/graphics/backend/vulkan/vulkan_shader.h"

#include "grassland/vulkan/spirv_nonuniform.h"

namespace grassland::graphics::backend {

VulkanShader::VulkanShader(VulkanCore *core, const CompiledShaderBlob &shader_blob) : core_(core) {
  if (shader_blob.data.size() % sizeof(uint32_t))
    throw std::invalid_argument("Invalid SPIR-V byte count");
  std::vector<uint32_t> words(shader_blob.data.size() / sizeof(uint32_t));
  std::memcpy(words.data(), shader_blob.data.data(), shader_blob.data.size());
  words = vulkan::RestoreStorageBufferNonUniform(words);
  vulkan::ThrowIfFailed(core_->Device()->CreateShaderModule(words.data(), words.size() * sizeof(uint32_t),
                                                            shader_blob.entry_point, &shader_module_),
                        "Create Vulkan shader module");
}

}  // namespace grassland::graphics::backend
