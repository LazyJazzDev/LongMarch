#include "grassland/graphics/backend/vulkan/vulkan_program.h"

namespace grassland::graphics::backend {

namespace {
std::vector<VkFormat> ConvertImageFormats(const std::vector<ImageFormat> &formats) {
  std::vector<VkFormat> result;
  for (auto format : formats) {
    result.push_back(ImageFormatToVkFormat(format));
  }
  return result;
}
}  // namespace

VulkanShader::VulkanShader(VulkanCore *core, const CompiledShaderBlob &shader_blob) : core_(core) {
  core_->Device()->CreateShaderModule(shader_blob, &shader_module_);
}

VulkanProgram::VulkanProgram(VulkanCore *core, const std::vector<ImageFormat> &color_formats, ImageFormat depth_format)
    : core_(core),
      pipeline_settings_(nullptr, ConvertImageFormats(color_formats), ImageFormatToVkFormat(depth_format)) {
  pipeline_settings_.EnableDynamicPrimitiveTopology();
}

VulkanProgram::~VulkanProgram() {
  pipeline_.reset();
  pipeline_layout_.reset();
  descriptor_set_layouts_.clear();
}

void VulkanProgram::AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) {
  pipeline_settings_.AddInputAttribute(binding, pipeline_settings_.vertex_input_attribute_descriptions.size(),
                                       InputTypeToVkFormat(type), offset);
}

void VulkanProgram::AddInputBinding(uint32_t stride, bool input_per_instance) {
  pipeline_settings_.AddInputBinding(pipeline_settings_.vertex_input_binding_descriptions.size(), stride,
                                     input_per_instance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX);
}

void VulkanProgram::AddResourceBinding(ResourceType type, int count) {
  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = ResourceTypeToVkDescriptorType(type);
  binding.descriptorCount = count;
  binding.stageFlags = VK_SHADER_STAGE_ALL;
  std::unique_ptr<vulkan::DescriptorSetLayout> descriptor_set_layout;
  core_->Device()->CreateDescriptorSetLayout({binding}, &descriptor_set_layout);
  descriptor_set_layouts_.push_back(std::move(descriptor_set_layout));
}

void VulkanProgram::SetCullMode(CullMode mode) {
  pipeline_settings_.SetCullMode(CullModeToVkCullMode(mode));
}

void VulkanProgram::SetBlendState(int target_id, const BlendState &state) {
  pipeline_settings_.SetBlendState(target_id, BlendStateToVkPipelineColorBlendAttachmentState(state));
}

void VulkanProgram::BindShader(Shader *shader, ShaderType type) {
  VulkanShader *vulkan_shader = dynamic_cast<VulkanShader *>(shader);
  if (vulkan_shader) {
    pipeline_settings_.AddShaderStage(vulkan_shader->ShaderModule(), ShaderTypeToVkShaderStageFlags(type),
                                      vulkan_shader->EntryPoint().c_str());
  } else {
    throw std::runtime_error("Invalid shader object, expected VulkanShader");
  }
}

void VulkanProgram::Finalize() {
  std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
  descriptor_set_layouts.reserve(descriptor_set_layouts_.size());
  for (auto &descriptor_set_layout : descriptor_set_layouts_) {
    descriptor_set_layouts.push_back(descriptor_set_layout->Handle());
  }
  core_->Device()->CreatePipelineLayout(descriptor_set_layouts, &pipeline_layout_);
  pipeline_settings_.pipeline_layout = pipeline_layout_.get();
  core_->Device()->CreatePipeline(pipeline_settings_, &pipeline_);
}

int VulkanProgram::NumInputBindings() const {
  return pipeline_settings_.vertex_input_binding_descriptions.size();
}

const vulkan::PipelineSettings *VulkanProgram::PipelineSettings() const {
  return &pipeline_settings_;
}

}  // namespace grassland::graphics::backend
