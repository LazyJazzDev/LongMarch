#include "grassland/graphics/backend/vulkan/vulkan_program.h"

namespace grassland::graphics::backend {

namespace {
std::vector<VkFormat> ConvertImageFormats(
    const std::vector<ImageFormat> &formats) {
  std::vector<VkFormat> result;
  for (auto format : formats) {
    result.push_back(ImageFormatToVkFormat(format));
  }
  return result;
}
}  // namespace

VulkanShader::VulkanShader(VulkanCore *core, const void *data, size_t size)
    : core_(core) {
  core_->Device()->CreateShaderModule(data, size, &shader_module_);
}

VulkanProgram::VulkanProgram(VulkanCore *core,
                             const std::vector<ImageFormat> &color_formats,
                             ImageFormat depth_format)
    : core_(core),
      pipeline_settings_(nullptr,
                         ConvertImageFormats(color_formats),
                         ImageFormatToVkFormat(depth_format)) {
}

VulkanProgram::~VulkanProgram() {
  pipeline_.reset();
  pipeline_layout_.reset();
  descriptor_set_layout_.reset();
}

void VulkanProgram::AddInputAttribute(uint32_t binding,
                                      InputType type,
                                      uint32_t offset) {
  pipeline_settings_.AddInputAttribute(
      binding, pipeline_settings_.vertex_input_attribute_descriptions.size(),
      InputTypeToVkFormat(type), offset);
}

void VulkanProgram::AddInputBinding(uint32_t stride, bool input_per_instance) {
  pipeline_settings_.AddInputBinding(
      pipeline_settings_.vertex_input_binding_descriptions.size(), stride,
      input_per_instance ? VK_VERTEX_INPUT_RATE_INSTANCE
                         : VK_VERTEX_INPUT_RATE_VERTEX);
}

void VulkanProgram::BindShader(Shader *shader, ShaderType type) {
  VulkanShader *vulkan_shader = dynamic_cast<VulkanShader *>(shader);
  if (vulkan_shader) {
    pipeline_settings_.AddShaderStage(vulkan_shader->ShaderModule(),
                                      ShaderTypeToVkShaderStageFlags(type));
  } else {
    throw std::runtime_error("Invalid shader object, expected VulkanShader");
  }
}

void VulkanProgram::Finalize() {
  core_->Device()->CreateDescriptorSetLayout({}, &descriptor_set_layout_);
  core_->Device()->CreatePipelineLayout({descriptor_set_layout_->Handle()},
                                        &pipeline_layout_);
  pipeline_settings_.pipeline_layout = pipeline_layout_.get();
  core_->Device()->CreatePipeline(pipeline_settings_, &pipeline_);
}
}  // namespace grassland::graphics::backend
