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

VulkanProgramBase::~VulkanProgramBase() {
  if (pipeline_layout_)
    vkDestroyPipelineLayout(core_->Handle(), pipeline_layout_, nullptr);
  for (VkDescriptorSetLayout layout : descriptor_set_layouts_) {
    vkDestroyDescriptorSetLayout(core_->Handle(), layout, nullptr);
  }
}

void VulkanProgramBase::AddResourceBindingImpl(ResourceType type, int count) {
  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = ResourceTypeToVkDescriptorType(type);
  binding.descriptorCount = count;
  binding.stageFlags = VK_SHADER_STAGE_ALL;
  VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  info.bindingCount = 1;
  info.pBindings = &binding;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  vulkan::ThrowIfFailed(vkCreateDescriptorSetLayout(core_->Handle(), &info, nullptr, &layout),
                        "Failed to create Vulkan descriptor set layout");
  descriptor_set_layouts_.push_back(layout);
  descriptor_bindings_.push_back(binding);
}

void VulkanProgramBase::FinalizePipelineLayout() {
  // Bindings currently use VK_SHADER_STAGE_ALL and ordinary descriptor sets.
  // Reject oversized scenes even when validation is disabled.
  const auto limits = vulkan::GetPhysicalDeviceProperties(core_->PhysicalDevice()).limits;
  uint64_t storage_buffers = 0, sampled_images = 0;
  for (const auto &binding : descriptor_bindings_) {
    if (binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      storage_buffers += binding.descriptorCount;
    if (binding.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
      sampled_images += binding.descriptorCount;
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
  descriptor_set_layouts = descriptor_set_layouts_;
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size());
  layout_info.pSetLayouts = descriptor_set_layouts.data();
  vulkan::ThrowIfFailed(vkCreatePipelineLayout(core_->Handle(), &layout_info, nullptr, &pipeline_layout_),
                        "Failed to create Vulkan pipeline layout");
}

VulkanProgram::VulkanProgram(VulkanCore *core, const std::vector<ImageFormat> &color_formats, ImageFormat depth_format)
    : VulkanProgramBase(core),
      pipeline_settings_(nullptr, ConvertImageFormats(color_formats), ImageFormatToVkFormat(depth_format)) {
  pipeline_settings_.EnableDynamicPrimitiveTopology();
}

VulkanProgram::~VulkanProgram() {
  if (pipeline_)
    vkDestroyPipeline(core_->Handle(), pipeline_, nullptr);
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
    pipeline_settings_.AddShaderStage(vulkan_shader->ModuleHandle(), vulkan_shader->EntryPointRef(),
                                      ShaderTypeToVkShaderStageFlags(type));
  } else {
    throw std::runtime_error("Invalid shader object, expected VulkanShader");
  }
}

void VulkanProgram::Finalize() {
  FinalizePipelineLayout();
  pipeline_settings_.pipeline_layout = pipeline_layout_;
  core_->CreatePipeline(pipeline_settings_, &pipeline_);
}

int VulkanProgram::NumInputBindings() const {
  return pipeline_settings_.vertex_input_binding_descriptions.size();
}

const vulkan::PipelineSettings *VulkanProgram::PipelineSettings() const {
  return &pipeline_settings_;
}

VulkanComputeProgram::VulkanComputeProgram(VulkanCore *core, VulkanShader *compute_shader)
    : VulkanProgramBase(core),
      compute_shader_(compute_shader) {
}

VulkanComputeProgram::~VulkanComputeProgram() {
  vkDestroyPipeline(core_->Handle(), pipeline_, nullptr);
}

void VulkanComputeProgram::AddResourceBinding(ResourceType type, int count) {
  AddResourceBindingImpl(type, count);
}

void VulkanComputeProgram::Finalize() {
  FinalizePipelineLayout();
  VkComputePipelineCreateInfo pipeline_create_info = {};
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.layout = pipeline_layout_;
  pipeline_create_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_create_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_create_info.stage.module = compute_shader_->ModuleHandle();
  pipeline_create_info.stage.pName = compute_shader_->EntryPointRef().c_str();
  pipeline_create_info.stage.pSpecializationInfo = nullptr;
  const VkResult result =
      vkCreateComputePipelines(core_->Handle(), VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline_);
  if (result != VK_SUCCESS)
    throw std::runtime_error("failed to create Vulkan compute pipeline: " + std::to_string(result));
}

VulkanRayTracingProgram::VulkanRayTracingProgram(VulkanCore *core) : VulkanProgramBase(core) {
}

VulkanRayTracingProgram::~VulkanRayTracingProgram() {
  if (sbt_buffer_)
    vmaDestroyBuffer(core_->Allocator(), sbt_buffer_, sbt_allocation_);
  if (pipeline_)
    vkDestroyPipeline(core_->Handle(), pipeline_, nullptr);
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
  raygen_shader_ = vk_raygen_shader;
}

void VulkanRayTracingProgram::AddMissShader(Shader *miss_shader) {
  auto vk_miss_shader = dynamic_cast<VulkanShader *>(miss_shader);
  assert(vk_miss_shader != nullptr);
  miss_shaders_.emplace_back(vk_miss_shader);
}

void VulkanRayTracingProgram::AddHitGroup(HitGroup hit_group) {
  vulkan::HitGroup vk_hit_group;
  auto vk_closest_hit_shader = dynamic_cast<VulkanShader *>(hit_group.closest_hit_shader);
  vk_hit_group.closest_hit_shader = vk_closest_hit_shader;
  assert(vk_hit_group.closest_hit_shader != nullptr);
  auto vk_any_hit_shader = dynamic_cast<VulkanShader *>(hit_group.any_hit_shader);
  if (vk_any_hit_shader) {
    vk_hit_group.any_hit_shader = vk_any_hit_shader;
  }

  auto vk_intersection_shader = dynamic_cast<VulkanShader *>(hit_group.intersection_shader);
  if (vk_intersection_shader) {
    vk_hit_group.intersection_shader = vk_intersection_shader;
  }
  vk_hit_group.procedure = hit_group.procedure;
  hit_groups_.emplace_back(std::move(vk_hit_group));
}

void VulkanRayTracingProgram::AddCallableShader(Shader *callable_shader) {
  auto vk_callable_shader = dynamic_cast<VulkanShader *>(callable_shader);
  assert(vk_callable_shader != nullptr);
  callable_shaders_.emplace_back(vk_callable_shader);
}

void VulkanRayTracingProgram::Finalize(const std::vector<int32_t> &miss_shader_indices,
                                       const std::vector<int32_t> &hit_group_indices,
                                       const std::vector<int32_t> &callable_shader_indices) {
  FinalizePipelineLayout();
  core_->CreateRayTracingPipeline(pipeline_layout_, raygen_shader_, miss_shaders_, hit_groups_, callable_shaders_,
                                  &pipeline_);
  vulkan::ThrowIfFailed(
      core_->CreateShaderBindingTable(pipeline_, miss_shaders_.size(), hit_groups_.size(), miss_shader_indices,
                                      hit_group_indices, callable_shader_indices, &sbt_buffer_, &sbt_allocation_,
                                      &raygen_address_, &miss_address_, &hit_address_, &callable_address_),
      "Failed to create Vulkan shader binding table");
  miss_shader_count_ = miss_shader_indices.size();
  hit_group_count_ = hit_group_indices.size();
  callable_shader_count_ = callable_shader_indices.size();
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
