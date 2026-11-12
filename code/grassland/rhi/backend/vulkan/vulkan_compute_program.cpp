#include "grassland/rhi/backend/vulkan/vulkan_compute_program.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_compute_program_impl.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanComputeProgramImpl::VulkanComputeProgramImpl(IVulkanDevice *device, const ComputeProgramDesc &desc) {
  InitializeComputeProgram(device, desc);
  InitializeVulkanProgramBase(device->NativeDevice());
}

VulkanComputeProgramImpl::~VulkanComputeProgramImpl() {
  auto device = LayoutDevice();
  if (pipeline_)
    vkDestroyPipeline(device, pipeline_, nullptr);

  DestroyVulkanProgramBase();
}

VkPipeline VulkanComputeProgramImpl::NativePipeline() const {
  return pipeline_;
}

rhi::IComputeProgram *VulkanComputeProgramImpl::Create(IVulkanDevice *owner_device, const ComputeProgramDesc &desc) {
  auto object = std::unique_ptr<VulkanComputeProgramImpl>(new VulkanComputeProgramImpl(owner_device, desc));
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

rhi::IComputeProgram *IVulkanComputeProgram::Create(IVulkanDevice *device, const ComputeProgramDesc &desc) {
  return VulkanComputeProgramImpl::Create(device, desc);
}

void VulkanComputeProgramImpl::InitializeComputeProgram(IVulkanDevice *device, const ComputeProgramDesc &desc) {
  device_ = device;
  desc_ = desc;
  shader_ = ObjectPtr<IShader>::Retain(desc.compute_shader);
}

const ComputeProgramDesc &VulkanComputeProgramImpl::GetDesc() const {
  return desc_;
}

IVulkanDevice *VulkanComputeProgramImpl::Owner() const {
  return device_;
}

bool VulkanComputeProgramImpl::ValidateLayout() const {
  if (!desc_.compute_shader || desc_.compute_shader->GetStage() != SHADER_STAGE_COMPUTE ||
      desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}

void VulkanComputeProgramImpl::InitializeVulkanProgramBase(VkDevice device) {
  layout_device_ = device;
}

void VulkanComputeProgramImpl::DestroyVulkanProgramBase() {
  if (layout_)
    vkDestroyPipelineLayout(layout_device_, layout_, nullptr);
  for (auto layout : set_layouts_)
    vkDestroyDescriptorSetLayout(layout_device_, layout, nullptr);
}

VkPipelineLayout VulkanComputeProgramImpl::NativeLayout() const {
  return layout_;
}

bool VulkanComputeProgramImpl::CreateLayout(const PipelineLayoutDesc &desc) {
  set_layouts_.reserve(desc.groups.size());
  for (const auto &group : desc.groups) {
    auto layout = CreateBindingLayout(layout_device_, group);
    if (!layout)
      return false;
    set_layouts_.push_back(layout);
  }
  VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  create.setLayoutCount = uint32_t(set_layouts_.size());
  create.pSetLayouts = set_layouts_.data();
  return Check(vkCreatePipelineLayout(layout_device_, &create, nullptr, &layout_), "Create pipeline layout");
}

VkDevice VulkanComputeProgramImpl::LayoutDevice() const {
  return layout_device_;
}
}  // namespace grassland::rhi::backend::vulkan
