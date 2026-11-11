#include "grassland/rhi/backend/vulkan/vulkan_program.h"

#include <unordered_set>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_impl.h"

namespace grassland::rhi::backend::vulkan {
VulkanProgramImpl::VulkanProgramImpl(IVulkanDevice *device, const ProgramDesc &desc) {
  InitializeProgram(device, desc);
  InitializeVulkanProgramBase(device->NativeDevice());
}

VulkanProgramImpl::~VulkanProgramImpl() {
  auto device = LayoutDevice();
  if (pipeline_)
    vkDestroyPipeline(device, pipeline_, nullptr);

  DestroyVulkanProgramBase();
}

VkPipeline VulkanProgramImpl::NativePipeline() const {
  return pipeline_;
}

rhi::IProgram *VulkanProgramImpl::Create(IVulkanDevice *owner_device, const ProgramDesc &desc) {
  auto object = std::unique_ptr<VulkanProgramImpl>(new VulkanProgramImpl(owner_device, desc));
  auto device = owner_device->NativeDevice();

  struct ShaderModules {
    VkDevice device;
    VkShaderModule handles[2]{};

    ~ShaderModules() {
      for (auto module : handles)
        if (module)
          vkDestroyShaderModule(device, module, nullptr);
    }
  } shader_modules{device};

  auto &modules = shader_modules.handles;
  if (!object->ValidateLayout() ||
      !ValidatePipelineBindings(owner_device->NativePhysicalDevice(), object->desc_.pipeline_layout)) {
    ReportError(-1, "Invalid graphics program layout.");
    return nullptr;
  }
  std::vector<uint8_t> code[2];
  auto prefix = ShaderMacroHeader(object->desc_.pipeline_layout, BACKEND_API_VULKAN, object->desc_.vertex_attributes);
  if (!CompileSlang(object->desc_.vertex_shader, prefix, BACKEND_API_VULKAN, code[0], &object->desc_.pipeline_layout))
    return nullptr;
  if (!CompileSlang(object->desc_.fragment_shader, prefix, BACKEND_API_VULKAN, code[1], &object->desc_.pipeline_layout))
    return nullptr;
  for (int i = 0; i < 2; ++i) {
    VkShaderModuleCreateInfo create{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    create.codeSize = code[i].size();
    create.pCode = reinterpret_cast<const uint32_t *>(code[i].data());
    if (!Check(vkCreateShaderModule(device, &create, nullptr, &modules[i]), "Create shader module")) {
      return nullptr;
    }
  }
  if (!object->CreateLayout(object->desc_.pipeline_layout))
    return nullptr;
  VkPipelineShaderStageCreateInfo stages[2]{};
  for (int i = 0; i < 2; ++i) {
    stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[i].module = modules[i];
  }
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].pName = object->desc_.vertex_shader->GetEntry().c_str();
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].pName = object->desc_.fragment_shader->GetEntry().c_str();
  std::vector<VkVertexInputBindingDescription> bindings;
  for (uint32_t i = 0; i < object->desc_.vertex_bindings.size(); ++i) {
    const auto &b = object->desc_.vertex_bindings[i];
    bindings.push_back({i, b.stride, b.per_instance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
  }
  const VkFormat formats[] = {VK_FORMAT_R32_SFLOAT,          VK_FORMAT_R32G32_SFLOAT,     VK_FORMAT_R32G32B32_SFLOAT,
                              VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32_SINT,          VK_FORMAT_R32G32_SINT,
                              VK_FORMAT_R32G32B32_SINT,      VK_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32_UINT,
                              VK_FORMAT_R32G32_UINT,         VK_FORMAT_R32G32B32_UINT,    VK_FORMAT_R32G32B32A32_UINT};
  std::vector<VkVertexInputAttributeDescription> attributes;
  for (const auto &a : object->desc_.vertex_attributes)
    attributes.push_back({a.location, a.binding, formats[a.type], a.offset});
  VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  input.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
  input.pVertexBindingDescriptions = bindings.data();
  input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
  input.pVertexAttributeDescriptions = attributes.data();
  VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  assembly.topology = NativePrimitiveTopology(desc.topology);
  VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = desc.rasterizer.fill_mode == FILL_MODE_WIREFRAME ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
  const VkCullModeFlags cull_modes[] = {VK_CULL_MODE_NONE, VK_CULL_MODE_FRONT_BIT, VK_CULL_MODE_BACK_BIT};
  raster.cullMode = cull_modes[desc.rasterizer.cull_mode];
  // Paired with the negative-height Vulkan viewport, as in graphics.
  raster.frontFace = desc.rasterizer.front_face == FRONT_FACE_COUNTER_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                                                : VK_FRONT_FACE_CLOCKWISE;
  raster.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo sample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  sample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  std::vector<VkPipelineColorBlendAttachmentState> blend_attachments;
  for (size_t i = 0; i < object->desc_.color_formats.size(); ++i) {
    const auto value = object->desc_.blend_states.empty() ? BlendState{} : object->desc_.blend_states[i];
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.blendEnable = value.blend_enable;
    attachment.srcColorBlendFactor = NativeBlendFactor(value.src_color);
    attachment.dstColorBlendFactor = NativeBlendFactor(value.dst_color);
    attachment.colorBlendOp = NativeBlendOp(value.color_op);
    attachment.srcAlphaBlendFactor = NativeBlendFactor(value.src_alpha, true);
    attachment.dstAlphaBlendFactor = NativeBlendFactor(value.dst_alpha, true);
    attachment.alphaBlendOp = NativeBlendOp(value.alpha_op);
    attachment.colorWriteMask = 0xf;
    blend_attachments.push_back(attachment);
  }
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(owner_device->NativePhysicalDevice(), &features);
  if (desc.rasterizer.fill_mode == FILL_MODE_WIREFRAME && !features.fillModeNonSolid) {
    ReportError(-1, "Wireframe rasterization is unsupported by this Vulkan device.");
    return nullptr;
  }
  if (!features.independentBlend && !blend_attachments.empty()) {
    const auto &first = blend_attachments.front();
    for (const auto &value : blend_attachments)
      if (value.blendEnable != first.blendEnable || value.srcColorBlendFactor != first.srcColorBlendFactor ||
          value.dstColorBlendFactor != first.dstColorBlendFactor || value.colorBlendOp != first.colorBlendOp ||
          value.srcAlphaBlendFactor != first.srcAlphaBlendFactor ||
          value.dstAlphaBlendFactor != first.dstAlphaBlendFactor || value.alphaBlendOp != first.alphaBlendOp) {
        ReportError(-1, "Independent color-attachment blending is unsupported.");
        return nullptr;
      }
  }
  VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  blend.attachmentCount = uint32_t(blend_attachments.size());
  blend.pAttachments = blend_attachments.data();
  const auto &ds = object->desc_.depth_stencil;
  VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth.depthTestEnable = ds.depth_test;
  depth.depthWriteEnable = ds.depth_write;
  depth.depthCompareOp = static_cast<VkCompareOp>(ds.depth_compare);
  depth.stencilTestEnable = ds.stencil_test;
  auto face = [&](const StencilFaceDesc &value) {
    VkStencilOpState result{};
    result.failOp = static_cast<VkStencilOp>(value.fail_op);
    result.depthFailOp = static_cast<VkStencilOp>(value.depth_fail_op);
    result.passOp = static_cast<VkStencilOp>(value.pass_op);
    result.compareOp = static_cast<VkCompareOp>(value.compare);
    result.compareMask = ds.stencil_read_mask;
    result.writeMask = ds.stencil_write_mask;
    result.reference = ds.stencil_reference;
    return result;
  };
  depth.front = face(ds.front);
  depth.back = face(ds.back);
  VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = 2;
  dynamic.pDynamicStates = dynamic_states;
  VkGraphicsPipelineCreateInfo create{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  create.stageCount = 2;
  create.pStages = stages;
  create.pVertexInputState = &input;
  create.pInputAssemblyState = &assembly;
  create.pViewportState = &viewport;
  create.pRasterizationState = &raster;
  create.pMultisampleState = &sample;
  create.pColorBlendState = &blend;
  create.pDepthStencilState = &depth;
  create.pDynamicState = &dynamic;
  create.layout = object->NativeLayout();
  std::vector<VkFormat> color_formats;
  for (auto format : object->desc_.color_formats)
    color_formats.push_back(NativeImageFormat(format));
  VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  if (owner_device->Procedures().CmdBeginRendering) {
    rendering.colorAttachmentCount = uint32_t(color_formats.size());
    rendering.pColorAttachmentFormats = color_formats.data();
    rendering.depthAttachmentFormat = NativeImageFormat(ds.format);
    rendering.stencilAttachmentFormat = HasStencil(ds.format) ? NativeImageFormat(ds.format) : VK_FORMAT_UNDEFINED;
    create.pNext = &rendering;
  } else {
    create.renderPass = owner_device->RenderTargets().RenderPass(object->desc_.color_formats, ds.format);
    if (!create.renderPass) {
      return nullptr;
    }
  }
  auto result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create, nullptr, &object->pipeline_);
  if (!Check(result, "Create graphics pipeline"))
    return nullptr;
  return object.release();
}

rhi::IProgram *IVulkanProgram::Create(IVulkanDevice *device, const ProgramDesc &desc) {
  return VulkanProgramImpl::Create(device, desc);
}

void VulkanProgramImpl::InitializeProgram(IBackendDevice *device, const ProgramDesc &desc) {
  device_ = device;
  desc_ = desc;
  vertex_shader_ = ObjectPtr<IShader>::Retain(desc.vertex_shader);
  fragment_shader_ = ObjectPtr<IShader>::Retain(desc.fragment_shader);
}

const ProgramDesc &VulkanProgramImpl::GetDesc() const {
  return desc_;
}

IBackendDevice *VulkanProgramImpl::Owner() const {
  return device_;
}

bool VulkanProgramImpl::ValidateLayout() const {
  if (desc_.topology < PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || desc_.topology > PRIMITIVE_TOPOLOGY_POINT_LIST ||
      desc_.rasterizer.cull_mode < CULL_MODE_NONE || desc_.rasterizer.cull_mode > CULL_MODE_BACK ||
      desc_.rasterizer.front_face < FRONT_FACE_COUNTER_CLOCKWISE ||
      desc_.rasterizer.front_face > FRONT_FACE_CLOCKWISE || desc_.rasterizer.fill_mode < FILL_MODE_SOLID ||
      desc_.rasterizer.fill_mode > FILL_MODE_WIREFRAME)
    return false;
  if (desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  if (!desc_.vertex_shader || !desc_.fragment_shader)
    return false;
  if (desc_.vertex_shader->GetStage() != SHADER_STAGE_VERTEX)
    return false;
  if (desc_.fragment_shader->GetStage() != SHADER_STAGE_FRAGMENT)
    return false;
  if (desc_.color_formats.size() > 8 || (desc_.color_formats.empty() && !IsDepthFormat(desc_.depth_stencil.format)))
    return false;
  for (auto format : desc_.color_formats)
    if (!IsColorFormat(format))
      return false;
  if (!desc_.blend_states.empty() && desc_.blend_states.size() != desc_.color_formats.size())
    return false;
  for (size_t i = 0; i < desc_.blend_states.size(); ++i) {
    if (desc_.blend_states[i].blend_enable &&
        (IsSignedIntegerFormat(desc_.color_formats[i]) || IsUnsignedIntegerFormat(desc_.color_formats[i])))
      return false;
  }
  for (const auto &blend : desc_.blend_states) {
    for (auto factor : {blend.src_color, blend.dst_color, blend.src_alpha, blend.dst_alpha})
      if (factor < BLEND_FACTOR_ZERO || factor > BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        return false;
    for (auto op : {blend.color_op, blend.alpha_op})
      if (op < BLEND_OP_ADD || op > BLEND_OP_MAX)
        return false;
  }
  const auto &depth = desc_.depth_stencil;
  if (depth.format != IMAGE_FORMAT_UNKNOWN && !IsDepthFormat(depth.format))
    return false;
  if ((depth.depth_test || depth.depth_write || depth.stencil_test) && !IsDepthFormat(depth.format))
    return false;
  if (depth.stencil_test && !HasStencil(depth.format))
    return false;
  if (depth.depth_compare < COMPARE_NEVER || depth.depth_compare > COMPARE_ALWAYS)
    return false;
  for (auto face : {depth.front, depth.back}) {
    if (face.compare < COMPARE_NEVER || face.compare > COMPARE_ALWAYS)
      return false;
    for (auto op : {face.fail_op, face.depth_fail_op, face.pass_op})
      if (op < STENCIL_KEEP || op > STENCIL_DECREMENT_WRAP)
        return false;
  }
  std::unordered_set<uint32_t> locations;
  for (const auto &attribute : desc_.vertex_attributes) {
    if (attribute.binding >= desc_.vertex_bindings.size())
      return false;
    auto size = InputTypeSize(attribute.type);
    auto stride = desc_.vertex_bindings[attribute.binding].stride;
    if (!size || attribute.offset > stride || size > stride - attribute.offset)
      return false;
    if (!locations.insert(attribute.location).second)
      return false;
  }
  return true;
}

void VulkanProgramImpl::InitializeVulkanProgramBase(VkDevice device) {
  layout_device_ = device;
}

void VulkanProgramImpl::DestroyVulkanProgramBase() {
  if (layout_)
    vkDestroyPipelineLayout(layout_device_, layout_, nullptr);
  for (auto layout : set_layouts_)
    vkDestroyDescriptorSetLayout(layout_device_, layout, nullptr);
}

VkPipelineLayout VulkanProgramImpl::NativeLayout() const {
  return layout_;
}

bool VulkanProgramImpl::CreateLayout(const PipelineLayoutDesc &desc) {
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

VkDevice VulkanProgramImpl::LayoutDevice() const {
  return layout_device_;
}
}  // namespace grassland::rhi::backend::vulkan
