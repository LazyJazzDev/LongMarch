#include "grassland/graphics/backend/vulkan/vulkan_program.h"

#include <numeric>
#include <stdexcept>

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
  // Bindings currently use VK_SHADER_STAGE_ALL and ordinary descriptor sets.
  // Reject oversized scenes even when validation is disabled.
  const auto limits = core_->Device()->PhysicalDevice().GetPhysicalDeviceProperties().limits;
  uint64_t storage_buffers = 0, sampled_images = 0;
  for (const auto &layout : descriptor_set_layouts_) {
    for (const auto &binding : layout->Bindings()) {
      if (binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        storage_buffers += binding.descriptorCount;
      if (binding.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
        sampled_images += binding.descriptorCount;
    }
  }
  if (descriptor_set_layouts_.size() > limits.maxBoundDescriptorSets ||
      storage_buffers > limits.maxPerStageDescriptorStorageBuffers ||
      sampled_images > limits.maxPerStageDescriptorSampledImages)
    throw std::runtime_error("pipeline exceeds Vulkan descriptor limits: " + std::to_string(storage_buffers) +
                             " storage buffers (limit " + std::to_string(limits.maxPerStageDescriptorStorageBuffers) +
                             "), " + std::to_string(sampled_images) + " sampled images (limit " +
                             std::to_string(limits.maxPerStageDescriptorSampledImages) + "), " +
                             std::to_string(descriptor_set_layouts_.size()) + " sets (limit " +
                             std::to_string(limits.maxBoundDescriptorSets) + ")");
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
  RecordResourceBinding(type, count);
  AddResourceBindingImpl(type, count);
}

void VulkanProgram::SetCullMode(CullMode mode) {
  pipeline_settings_.SetCullMode(CullModeToVkCullMode(mode));
}

void VulkanProgram::SetBlendState(int target_id, const BlendState &state) {
  pipeline_settings_.SetBlendState(target_id, BlendStateToVkPipelineColorBlendAttachmentState(state));
}

void VulkanProgram::BindShader(Shader *shader, ShaderType type) {
  if (!shader)
    throw std::invalid_argument("missing graphics shader");
  shader_stages_[type] = shader;
}

void VulkanProgram::Finalize() {
  pipeline_settings_.shader_stage_create_infos.clear();
  for (auto [type, shader] : shader_stages_) {
    VulkanShader *vulkan_shader = dynamic_cast<VulkanShader *>(ResolveShader(core_, shader));
    if (vulkan_shader) {
      pipeline_settings_.AddShaderStage(vulkan_shader->ShaderModule(), ShaderTypeToVkShaderStageFlags(type));
    } else {
      throw std::runtime_error("Invalid shader object, expected VulkanShader");
    }
  }
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

VulkanComputeProgram::VulkanComputeProgram(VulkanCore *core, Shader *compute_shader)
    : VulkanProgramBase(core),
      compute_shader_(compute_shader) {
}

VulkanComputeProgram::~VulkanComputeProgram() {
  vkDestroyPipeline(core_->Device()->Handle(), pipeline_, nullptr);
}

void VulkanComputeProgram::AddResourceBinding(ResourceType type, int count) {
  RecordResourceBinding(type, count);
  AddResourceBindingImpl(type, count);
}

void VulkanComputeProgram::Finalize() {
  auto shader = dynamic_cast<VulkanShader *>(ResolveShader(core_, compute_shader_));
  if (!shader)
    throw std::invalid_argument("invalid compute shader");
  FinalizePipelineLayout();
  VkComputePipelineCreateInfo pipeline_create_info = {};
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.layout = pipeline_layout_->Handle();
  pipeline_create_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_create_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_create_info.stage.module = shader->ShaderModule()->Handle();
  pipeline_create_info.stage.pName = shader->ShaderModule()->EntryPoint().c_str();
  pipeline_create_info.stage.pSpecializationInfo = nullptr;
  const VkResult result = vkCreateComputePipelines(core_->Device()->Handle(), VK_NULL_HANDLE, 1, &pipeline_create_info,
                                                   nullptr, &pipeline_);
  if (result != VK_SUCCESS)
    throw std::runtime_error("failed to create Vulkan compute pipeline: " + std::to_string(result));
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
  RecordResourceBinding(type, count);
  AddResourceBindingImpl(type, count);
}

void VulkanRayTracingProgram::AddRayGenShader(Shader *shader) {
  source_raygen_ = shader;
}

void VulkanRayTracingProgram::AddMissShader(Shader *shader) {
  source_miss_.push_back(shader);
}

void VulkanRayTracingProgram::AddHitGroup(HitGroup group) {
  source_hit_groups_.push_back(group);
}

void VulkanRayTracingProgram::AddCallableShader(Shader *shader) {
  source_callable_.push_back(shader);
}

void VulkanRayTracingProgram::ResolveShaders() {
  auto resolve = [&](Shader *source) -> vulkan::ShaderModule * {
    auto shader = dynamic_cast<VulkanShader *>(ResolveShader(core_, source));
    if (!shader)
      throw std::invalid_argument("invalid ray tracing shader");
    return shader->ShaderModule();
  };
  raygen_shader_ = resolve(source_raygen_);
  miss_shaders_.clear();
  hit_groups_.clear();
  callable_shaders_.clear();
  for (auto shader : source_miss_)
    miss_shaders_.push_back(resolve(shader));
  for (auto source : source_hit_groups_) {
    vulkan::HitGroup group{};
    group.closest_hit_shader = source.closest_hit_shader ? resolve(source.closest_hit_shader) : nullptr;
    group.any_hit_shader = source.any_hit_shader ? resolve(source.any_hit_shader) : nullptr;
    group.intersection_shader = source.intersection_shader ? resolve(source.intersection_shader) : nullptr;
    group.procedure = source.procedure;
    hit_groups_.push_back(group);
  }
  for (auto shader : source_callable_)
    callable_shaders_.push_back(resolve(shader));
}

void VulkanRayTracingProgram::Finalize(const std::vector<int32_t> &miss_shader_indices,
                                       const std::vector<int32_t> &hit_group_indices,
                                       const std::vector<int32_t> &callable_shader_indices) {
  ResolveShaders();
  FinalizePipelineLayout();
  core_->Device()->CreateRayTracingPipeline(pipeline_layout_.get(), raygen_shader_, miss_shaders_, hit_groups_,
                                            callable_shaders_, &pipeline_);
  core_->Device()->CreateShaderBindingTable(pipeline_.get(), miss_shader_indices, hit_group_indices,
                                            callable_shader_indices, &shader_binding_table_);
}

void VulkanRayTracingProgram::Finalize() {
  std::vector<int32_t> miss_shader_indices(source_miss_.size());
  std::iota(miss_shader_indices.begin(), miss_shader_indices.end(), 0);
  std::vector<int32_t> hit_group_indices(source_hit_groups_.size());
  std::iota(hit_group_indices.begin(), hit_group_indices.end(), 0);
  std::vector<int32_t> callable_shader_indices(source_callable_.size());
  std::iota(callable_shader_indices.begin(), callable_shader_indices.end(), 0);
  Finalize(miss_shader_indices, hit_group_indices, callable_shader_indices);
}

}  // namespace grassland::graphics::backend
