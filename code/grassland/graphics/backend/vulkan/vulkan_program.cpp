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

VulkanProgramBase::VulkanProgramBase(VulkanCore *core) : core_(core) {
}

void VulkanProgramBase::AddResourceBindingImpl(ResourceType type, int count) {
  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = ResourceTypeToVkDescriptorType(type);
  binding.descriptorCount = count;
  binding.stageFlags = VK_SHADER_STAGE_ALL;
  std::unique_ptr<vulkan::DescriptorSetLayout> descriptor_set_layout;
  core_->Device()->CreateDescriptorSetLayout({binding}, &descriptor_set_layout);
  descriptor_set_layouts_.push_back(std::move(descriptor_set_layout));
}

void VulkanProgramBase::FinalizePipelineLayout() {
  std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
  descriptor_set_layouts.reserve(descriptor_set_layouts_.size());
  for (auto &descriptor_set_layout : descriptor_set_layouts_) {
    descriptor_set_layouts.push_back(descriptor_set_layout->Handle());
  }
  core_->Device()->CreatePipelineLayout(descriptor_set_layouts, &pipeline_layout_);
}

VulkanProgram::VulkanProgram(VulkanCore *core, const std::vector<ImageFormat> &color_formats, ImageFormat depth_format)
    : VulkanProgramBase(core),
      pipeline_settings_(nullptr, ConvertImageFormats(color_formats), ImageFormatToVkFormat(depth_format)) {
  pipeline_settings_.EnableDynamicPrimitiveTopology();
}

VulkanProgram::~VulkanProgram() {
  pipeline_.reset();
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
  AddResourceBindingImpl(type, count);
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
  FinalizePipelineLayout();
  pipeline_settings_.pipeline_layout = pipeline_layout_.get();
  core_->Device()->CreatePipeline(pipeline_settings_, &pipeline_);
}

int VulkanProgram::NumInputBindings() const {
  return pipeline_settings_.vertex_input_binding_descriptions.size();
}

const vulkan::PipelineSettings *VulkanProgram::PipelineSettings() const {
  return &pipeline_settings_;
}

VulkanComputeProgram::VulkanComputeProgram(VulkanCore *core, VulkanShader *compute_shader)
    : VulkanProgramBase(core), compute_shader_(compute_shader) {
}

VulkanComputeProgram::~VulkanComputeProgram() {
  vkDestroyPipeline(core_->Device()->Handle(), pipeline_, nullptr);
}

void VulkanComputeProgram::AddResourceBinding(ResourceType type, int count) {
  AddResourceBindingImpl(type, count);
}

void VulkanComputeProgram::Finalize() {
  FinalizePipelineLayout();
  VkComputePipelineCreateInfo pipeline_create_info = {};
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.layout = pipeline_layout_->Handle();
  pipeline_create_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_create_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_create_info.stage.module = compute_shader_->ShaderModule()->Handle();
  pipeline_create_info.stage.pName = compute_shader_->EntryPoint().c_str();
  pipeline_create_info.stage.pSpecializationInfo = nullptr;
  vkCreateComputePipelines(core_->Device()->Handle(), VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline_);
}

VulkanRayTracingProgram::VulkanRayTracingProgram(VulkanCore *core) : VulkanProgramBase(core) {
}

VulkanRayTracingProgram::VulkanRayTracingProgram(VulkanCore *core,
                                                 VulkanShader *raygen_shader,
                                                 VulkanShader *miss_shader,
                                                 VulkanShader *closest_hit_shader)
    : VulkanRayTracingProgram(core) {
  AddRayGenShader(raygen_shader);
  AddMissShader(miss_shader);
  AddHitGroup({closest_hit_shader, nullptr, nullptr, false});
}

void VulkanRayTracingProgram::AddResourceBinding(ResourceType type, int count) {
  AddResourceBindingImpl(type, count);
}

void VulkanRayTracingProgram::AddRayGenShader(Shader *ray_gen_shader) {
  auto vk_raygen_shader = dynamic_cast<VulkanShader *>(ray_gen_shader);
  assert(vk_raygen_shader != nullptr);
  raygen_shader_ = vk_raygen_shader->ShaderModule();
}

void VulkanRayTracingProgram::AddMissShader(Shader *miss_shader) {
  auto vk_miss_shader = dynamic_cast<VulkanShader *>(miss_shader);
  assert(vk_miss_shader != nullptr);
  miss_shaders_.emplace_back(vk_miss_shader->ShaderModule());
}

void VulkanRayTracingProgram::AddHitGroup(HitGroup hit_group) {
  vulkan::HitGroup vk_hit_group;
  auto vk_closest_hit_shader = dynamic_cast<VulkanShader *>(hit_group.closest_hit_shader);
  vk_hit_group.closest_hit_shader = vk_closest_hit_shader->ShaderModule();
  assert(vk_hit_group.closest_hit_shader != nullptr);
  auto vk_any_hit_shader = dynamic_cast<VulkanShader *>(hit_group.any_hit_shader);
  if (vk_any_hit_shader) {
    vk_hit_group.any_hit_shader = vk_any_hit_shader->ShaderModule();
  }
  auto vk_intersection_shader = dynamic_cast<VulkanShader *>(hit_group.intersection_shader);
  if (vk_intersection_shader) {
    vk_hit_group.intersection_shader = vk_intersection_shader->ShaderModule();
  }
  vk_hit_group.procedure = hit_group.procedure;
  hit_groups_.emplace_back(std::move(vk_hit_group));
}

void VulkanRayTracingProgram::AddCallableShader(Shader *callable_shader) {
  auto vk_callable_shader = dynamic_cast<VulkanShader *>(callable_shader);
  assert(vk_callable_shader != nullptr);
  callable_shaders_.emplace_back(vk_callable_shader->ShaderModule());
}

void VulkanRayTracingProgram::Finalize(const std::vector<int32_t> &miss_shader_indices,
                                       const std::vector<int32_t> &hit_group_indices,
                                       const std::vector<int32_t> &callable_shader_indices) {
  FinalizePipelineLayout();
  core_->Device()->CreateRayTracingPipeline(pipeline_layout_.get(), raygen_shader_, miss_shaders_, hit_groups_,
                                            callable_shaders_, &pipeline_);
  core_->Device()->CreateShaderBindingTable(pipeline_.get(), miss_shader_indices, hit_group_indices,
                                            callable_shader_indices, &shader_binding_table_);
}

void VulkanRayTracingProgram::Finalize() {
  std::vector<int32_t> miss_shader_indices(miss_shaders_.size());
  std::iota(miss_shader_indices.begin(), miss_shader_indices.end(), 0);
  std::vector<int32_t> hit_group_indices(hit_groups_.size());
  std::iota(hit_group_indices.begin(), hit_group_indices.end(), 0);
  std::vector<int32_t> callable_shader_indices(callable_shaders_.size());
  std::iota(callable_shader_indices.begin(), callable_shader_indices.end(), 0);
  Finalize(miss_shader_indices, hit_group_indices, callable_shader_indices);
}

}  // namespace grassland::graphics::backend
