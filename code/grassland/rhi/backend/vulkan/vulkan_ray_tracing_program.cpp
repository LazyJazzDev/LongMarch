#include "grassland/rhi/backend/vulkan/vulkan_ray_tracing_program.h"

#include <cstring>

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanRayTracingProgram::VulkanRayTracingProgram(VulkanDevice *device, const RayTracingProgramDesc &desc)
    : RayTracingProgram(device, desc),
      VulkanProgramBase(device->NativeDevice()) {
}

VulkanRayTracingProgram::~VulkanRayTracingProgram() {
  if (pipeline_)
    vkDestroyPipeline(static_cast<VulkanDevice *>(device_)->NativeDevice(), pipeline_, nullptr);
}

VkPipeline VulkanRayTracingProgram::NativePipeline() const {
  return pipeline_;
}

const std::array<VkStridedDeviceAddressRegionKHR, 4> &VulkanRayTracingProgram::ShaderTables() const {
  return tables_;
}

bool VulkanRayTracingProgram::Finalize() {
  if (finalized_)
    return true;
  auto owner = static_cast<VulkanDevice *>(device_);
  auto device = owner->NativeDevice();
  if (!ValidateLayout() || !ValidatePipelineBindings(owner->NativePhysicalDevice(), desc_.pipeline_layout))
    return false;
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 physical{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &properties};
  vkGetPhysicalDeviceProperties2(owner->NativePhysicalDevice(), &physical);
  if (desc_.max_recursion_depth > properties.maxRayRecursionDepth ||
      desc_.max_attribute_size > properties.maxRayHitAttributeSize)
    return false;
  if (!CreateLayout(desc_.pipeline_layout))
    return false;
  std::vector<IShader *> shaders;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
  auto group = [] {
    VkRayTracingShaderGroupCreateInfoKHR value{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    value.generalShader = value.closestHitShader = value.anyHitShader = value.intersectionShader = VK_SHADER_UNUSED_KHR;
    return value;
  };
  auto general = [&](IShader *shader) {
    auto value = group();
    value.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    value.generalShader = uint32_t(shaders.size());
    shaders.push_back(shader);
    groups.push_back(value);
  };
  general(desc_.ray_generation);
  for (auto shader : desc_.miss_shaders)
    general(shader);
  for (auto hit : desc_.hit_groups) {
    auto value = group();
    value.type = hit.intersection ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                                  : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    if (hit.closest_hit) {
      value.closestHitShader = uint32_t(shaders.size());
      shaders.push_back(hit.closest_hit);
    }
    if (hit.any_hit) {
      value.anyHitShader = uint32_t(shaders.size());
      shaders.push_back(hit.any_hit);
    }
    if (hit.intersection) {
      value.intersectionShader = uint32_t(shaders.size());
      shaders.push_back(hit.intersection);
    }
    groups.push_back(value);
  }
  for (auto shader : desc_.callable_shaders)
    general(shader);
  std::vector<std::vector<uint8_t>> code(shaders.size());
  auto header = ShaderMacroHeader(desc_.pipeline_layout, BACKEND_API_VULKAN);
  for (size_t i = 0; i < shaders.size(); ++i)
    if (!CompileSlang(shaders[i]->GetDesc(), header, BACKEND_API_VULKAN, code[i], &desc_.pipeline_layout))
      return false;
  std::vector<VkPipelineShaderStageCreateInfo> stages(shaders.size());
  constexpr VkShaderStageFlagBits native_stages[] = {
      VK_SHADER_STAGE_VERTEX_BIT,      VK_SHADER_STAGE_FRAGMENT_BIT,         VK_SHADER_STAGE_COMPUTE_BIT,
      VK_SHADER_STAGE_RAYGEN_BIT_KHR,  VK_SHADER_STAGE_MISS_BIT_KHR,         VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
      VK_SHADER_STAGE_ANY_HIT_BIT_KHR, VK_SHADER_STAGE_INTERSECTION_BIT_KHR, VK_SHADER_STAGE_CALLABLE_BIT_KHR};
  bool success = true;
  for (size_t i = 0; i < shaders.size(); ++i) {
    auto &stage = stages[i];
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = native_stages[shaders[i]->GetDesc().stage];
    stage.pName = shaders[i]->GetDesc().entry_point.c_str();
    VkShaderModuleCreateInfo create{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    create.codeSize = code[i].size();
    create.pCode = reinterpret_cast<const uint32_t *>(code[i].data());
    if (!Check(vkCreateShaderModule(device, &create, nullptr, &stage.module), "Create ray tracing shader")) {
      success = false;
      break;
    }
  }
  VkPipeline pipeline{};
  if (success) {
    VkRayTracingPipelineCreateInfoKHR create{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    create.stageCount = uint32_t(stages.size());
    create.pStages = stages.data();
    create.groupCount = uint32_t(groups.size());
    create.pGroups = groups.data();
    create.maxPipelineRayRecursionDepth = desc_.max_recursion_depth;
    create.layout = NativeLayout();
    auto command = owner->Procedures().CreateRayTracingPipelinesKHR;
    success = Check(command(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &create, nullptr, &pipeline),
                    "Create ray tracing pipeline");
  }
  for (auto stage : stages)
    if (stage.module)
      vkDestroyShaderModule(device, stage.module, nullptr);
  if (!success)
    return false;
  // Keep ownership even when table allocation fails; a retry releases this pipeline first.
  if (pipeline_)
    vkDestroyPipeline(device, pipeline_, nullptr);
  pipeline_ = pipeline;
  const uint64_t handle_size = properties.shaderGroupHandleSize;
  const uint64_t stride = (handle_size + properties.shaderGroupHandleAlignment - 1) &
                          ~(uint64_t(properties.shaderGroupHandleAlignment) - 1);
  const uint64_t alignment = properties.shaderGroupBaseAlignment;
  if (stride > properties.maxShaderGroupStride)
    return false;
  std::vector<uint8_t> handles(groups.size() * handle_size);
  auto get_handles = owner->Procedures().GetRayTracingShaderGroupHandlesKHR;
  if (!Check(get_handles(device, pipeline, 0, uint32_t(groups.size()), handles.size(), handles.data()),
             "Get ray tracing shader handles"))
    return false;
  const size_t counts[] = {1, desc_.miss_shaders.size(), desc_.hit_groups.size(), desc_.callable_shaders.size()};
  uint64_t capacity = alignment;
  for (auto count : counts)
    capacity += count * stride + alignment;
  auto table = ObjectPtr<VulkanAllocatedBuffer>::Adopt(VulkanAllocatedBuffer::Create(owner, {capacity}, true));
  if (!table)
    return false;
  VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  address_info.buffer = table->NativeBuffer();
  auto address = vkGetBufferDeviceAddress(device, &address_info);
  std::vector<uint8_t> data(capacity);
  uint64_t offset = 0, handle = 0;
  for (size_t t = 0; t < 4; ++t) {
    offset = ((address + offset + alignment - 1) & ~(alignment - 1)) - address;
    if (counts[t])
      tables_[t] = {address + offset, stride, counts[t] * stride};
    else
      tables_[t] = {};
    for (size_t i = 0; i < counts[t]; ++i) {
      std::memcpy(data.data() + offset + i * stride, handles.data() + handle * handle_size, handle_size);
      ++handle;
    }
    offset += counts[t] * stride;
  }
  if (!table->Write(data.data(), data.size()))
    return false;
  table_ = std::move(table);
  finalized_ = true;
  return true;
}
}  // namespace grassland::rhi::backend::vulkan
