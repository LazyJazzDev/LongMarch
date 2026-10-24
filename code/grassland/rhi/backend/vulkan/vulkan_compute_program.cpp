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

VulkanComputeProgram *VulkanComputeProgram::Create(VulkanDevice *owner_device, const ComputeProgramDesc &desc) {
  auto object = std::unique_ptr<VulkanComputeProgram>(new VulkanComputeProgram(owner_device, desc));
  auto device = owner_device->NativeDevice();
  if (!object->ValidateLayout() ||
      !ValidatePipelineBindings(owner_device->NativePhysicalDevice(), object->desc_.pipeline_layout))
    return nullptr;
  std::vector<uint8_t> code;
  std::array<uint32_t, 3> threads{};
  if (!CompileSlang(object->desc_.compute_shader, ShaderMacroHeader(object->desc_.pipeline_layout, BACKEND_API_VULKAN),
                    BACKEND_API_VULKAN, code, &object->desc_.pipeline_layout, &threads))
    return nullptr;
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(owner_device->NativePhysicalDevice(), &properties);
  for (int i = 0; i < 3; ++i)
    if (!threads[i] || threads[i] > properties.limits.maxComputeWorkGroupSize[i])
      return nullptr;
  if (uint64_t(threads[0]) * threads[1] * threads[2] > properties.limits.maxComputeWorkGroupInvocations)
    return nullptr;
  if (!object->CreateLayout(object->desc_.pipeline_layout))
    return nullptr;

  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = code.size();
  module_info.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module{};
  if (!Check(vkCreateShaderModule(device, &module_info, nullptr, &module), "Create compute shader module"))
    return nullptr;
  VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  create.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  create.stage.module = module;
  create.stage.pName = object->desc_.compute_shader->GetEntry().c_str();
  create.layout = object->NativeLayout();
  auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create, nullptr, &object->pipeline_);
  vkDestroyShaderModule(device, module, nullptr);
  if (!Check(result, "Create compute pipeline"))
    return nullptr;
  return object.release();
}
}  // namespace grassland::rhi::backend::vulkan
