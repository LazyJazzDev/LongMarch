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

const ShaderBindingTable *VulkanRayTracingProgram::DefaultShaderBindingTable() const {
  return table_.Get();
}

IShaderBindingTable *VulkanRayTracingProgram::CreateShaderBindingTable(const ShaderBindingTableDesc &desc) {
  if (!ValidateShaderBindingTable(desc))
    return nullptr;
  return BuildShaderBindingTable(desc);
}

VulkanRayTracingProgram *VulkanRayTracingProgram::Create(VulkanDevice *owner_device,
                                                         const RayTracingProgramDesc &desc) {
  auto object = std::unique_ptr<VulkanRayTracingProgram>(new VulkanRayTracingProgram(owner_device, desc));
  auto device = owner_device->NativeDevice();
  if (!object->ValidateLayout() ||
      !ValidatePipelineBindings(owner_device->NativePhysicalDevice(), object->desc_.pipeline_layout))
    return nullptr;
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 physical{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &properties};
  vkGetPhysicalDeviceProperties2(owner_device->NativePhysicalDevice(), &physical);
  if (object->desc_.max_recursion_depth > properties.maxRayRecursionDepth ||
      object->desc_.max_attribute_size > properties.maxRayHitAttributeSize)
    return nullptr;
  if (!object->CreateLayout(object->desc_.pipeline_layout))
    return nullptr;
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
  general(object->desc_.ray_generation);
  for (auto shader : object->desc_.miss_shaders)
    general(shader);
  for (auto hit : object->desc_.hit_groups) {
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
  for (auto shader : object->desc_.callable_shaders)
    general(shader);
  std::vector<std::vector<uint8_t>> code(shaders.size());
  auto header = ShaderMacroHeader(object->desc_.pipeline_layout, BACKEND_API_VULKAN);
  for (size_t i = 0; i < shaders.size(); ++i)
    if (!CompileSlang(shaders[i], header, BACKEND_API_VULKAN, code[i], &object->desc_.pipeline_layout))
      return nullptr;
  std::vector<VkPipelineShaderStageCreateInfo> stages(shaders.size());
  constexpr VkShaderStageFlagBits native_stages[] = {
      VK_SHADER_STAGE_VERTEX_BIT,      VK_SHADER_STAGE_FRAGMENT_BIT,         VK_SHADER_STAGE_COMPUTE_BIT,
      VK_SHADER_STAGE_RAYGEN_BIT_KHR,  VK_SHADER_STAGE_MISS_BIT_KHR,         VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
      VK_SHADER_STAGE_ANY_HIT_BIT_KHR, VK_SHADER_STAGE_INTERSECTION_BIT_KHR, VK_SHADER_STAGE_CALLABLE_BIT_KHR};
  bool success = true;
  for (size_t i = 0; i < shaders.size(); ++i) {
    auto &stage = stages[i];
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = native_stages[shaders[i]->GetStage()];
    stage.pName = shaders[i]->GetEntry().c_str();
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
    create.maxPipelineRayRecursionDepth = object->desc_.max_recursion_depth;
    create.layout = object->NativeLayout();
    auto command = owner_device->Procedures().CreateRayTracingPipelinesKHR;
    success = Check(command(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &create, nullptr, &pipeline),
                    "Create ray tracing pipeline");
  }
  for (auto stage : stages)
    if (stage.module)
      vkDestroyShaderModule(device, stage.module, nullptr);
  // Own any returned pipeline before handling failure or allocating the default SBT.
  object->pipeline_ = pipeline;
  if (!success)
    return nullptr;
  object->handle_size_ = properties.shaderGroupHandleSize;
  object->stride_ = (object->handle_size_ + properties.shaderGroupHandleAlignment - 1) &
                    ~(uint64_t(properties.shaderGroupHandleAlignment) - 1);
  object->alignment_ = properties.shaderGroupBaseAlignment;
  if (object->stride_ > properties.maxShaderGroupStride)
    return nullptr;
  std::vector<uint8_t> handles(groups.size() * object->handle_size_);
  auto get_handles = owner_device->Procedures().GetRayTracingShaderGroupHandlesKHR;
  if (!Check(get_handles(device, pipeline, 0, uint32_t(groups.size()), handles.size(), handles.data()),
             "Get ray tracing shader handles"))
    return nullptr;
  const size_t counts[] = {1, object->desc_.miss_shaders.size(), object->desc_.hit_groups.size(),
                           object->desc_.callable_shaders.size()};
  size_t offset = 0;
  for (size_t t = 0; t < 4; ++t) {
    auto size = counts[t] * object->handle_size_;
    object->identifiers_[t].assign(handles.begin() + offset, handles.begin() + offset + size);
    offset += size;
  }
  object->table_ =
      ObjectPtr<ShaderBindingTable>::Adopt(object->BuildShaderBindingTable(object->DefaultShaderBindingTableDesc()));
  if (!object->table_)
    return nullptr;
  return object.release();
}

ShaderBindingTable *VulkanRayTracingProgram::BuildShaderBindingTable(const ShaderBindingTableDesc &desc) {
  const std::vector<uint32_t> raygen{0};
  const std::vector<uint32_t> *indices[] = {&raygen, &desc.miss_shader_indices, &desc.hit_group_indices,
                                            &desc.callable_shader_indices};
  uint64_t capacity = alignment_;
  for (auto table : indices)
    capacity += table->size() * stride_ + alignment_;
  auto owner = static_cast<VulkanDevice *>(device_);
  auto buffer = ObjectPtr<VulkanAllocatedBuffer>::Adopt(VulkanAllocatedBuffer::Create(owner, {capacity}, true));
  if (!buffer)
    return nullptr;
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  info.buffer = buffer->NativeBuffer();
  auto address = vkGetBufferDeviceAddress(owner->NativeDevice(), &info);
  std::vector<uint8_t> data(capacity);
  std::array<ShaderBindingTable::Region, 4> regions{};
  uint64_t offset = 0;
  for (size_t t = 0; t < 4; ++t) {
    if (indices[t]->empty())
      continue;
    offset = ((address + offset + alignment_ - 1) & ~(alignment_ - 1)) - address;
    regions[t] = {address + offset, indices[t]->size() * stride_, stride_};
    for (size_t i = 0; i < indices[t]->size(); ++i)
      std::memcpy(data.data() + offset + i * stride_, identifiers_[t].data() + (*indices[t])[i] * handle_size_,
                  handle_size_);
    offset += regions[t].size;
  }
  if (!buffer->Write(data.data(), data.size()))
    return nullptr;
  return new ShaderBindingTable(this, desc, std::move(buffer), regions);
}

}  // namespace grassland::rhi::backend::vulkan
