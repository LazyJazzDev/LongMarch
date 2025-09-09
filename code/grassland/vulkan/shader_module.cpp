#include "grassland/vulkan/shader_module.h"
namespace grassland::vulkan {

ShaderModule::ShaderModule(const struct Device *device, VkShaderModule shader_module, const std::string &entry_point)
    : device_(device), shader_module_(shader_module), entry_point_(entry_point) {
}

ShaderModule::~ShaderModule() {
  vkDestroyShaderModule(device_->Handle(), shader_module_, nullptr);
}

}  // namespace grassland::vulkan
