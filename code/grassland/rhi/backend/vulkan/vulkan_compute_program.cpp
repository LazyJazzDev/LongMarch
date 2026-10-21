#include "grassland/rhi/backend/vulkan/vulkan_compute_program.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanComputeProgram::VulkanComputeProgram(VulkanDevice *device, const ComputeProgramDesc &desc)
    : ComputeProgram(device, desc),
      VulkanProgramBase(device->NativeDevice()) {
}

VulkanComputeProgram::~VulkanComputeProgram() {
  auto device = static_cast<VulkanDevice *>(device_)->NativeDevice();
  if (pipeline_)
    vkDestroyPipeline(device, pipeline_, nullptr);
}

VkPipeline VulkanComputeProgram::NativePipeline() const {
  return pipeline_;
}

bool VulkanComputeProgram::Finalize() {
  if (finalized_)
    return true;
  auto owner = static_cast<VulkanDevice *>(device_);
  auto device = owner->NativeDevice();
  if (!ValidateLayout() || !ValidatePipelineBindings(owner->NativePhysicalDevice(), desc_.pipeline_layout))
    return false;
  std::vector<uint8_t> code;
  std::array<uint32_t, 3> threads{};
  if (!CompileSlang(desc_.compute_shader->GetDesc(), ShaderMacroHeader(desc_.pipeline_layout, BACKEND_API_VULKAN),
                    BACKEND_API_VULKAN, code, &desc_.pipeline_layout, &threads))
    return false;
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(owner->NativePhysicalDevice(), &properties);
  for (int i = 0; i < 3; ++i)
    if (!threads[i] || threads[i] > properties.limits.maxComputeWorkGroupSize[i])
      return false;
  if (uint64_t(threads[0]) * threads[1] * threads[2] > properties.limits.maxComputeWorkGroupInvocations)
    return false;
  if (!CreateLayout(desc_.pipeline_layout))
    return false;

  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = code.size();
  module_info.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module{};
  if (!Check(vkCreateShaderModule(device, &module_info, nullptr, &module), "Create compute shader module"))
    return false;
  VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  create.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  create.stage.module = module;
  create.stage.pName = desc_.compute_shader->GetDesc().entry_point.c_str();
  create.layout = NativeLayout();
  auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create, nullptr, &pipeline_);
  vkDestroyShaderModule(device, module, nullptr);
  if (!Check(result, "Create compute pipeline"))
    return false;
  finalized_ = true;
  return true;
}
}  // namespace grassland::rhi::backend::vulkan
