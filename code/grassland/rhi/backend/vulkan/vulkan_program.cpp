#include "grassland/rhi/backend/vulkan/vulkan_program.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanProgram::VulkanProgram(VulkanDevice *device, const ProgramDesc &desc)
    : Program(device, desc),
      VulkanProgramBase(device->NativeDevice()) {
}

VulkanProgram::~VulkanProgram() {
  auto device = static_cast<VulkanDevice *>(device_)->NativeDevice();
  if (pipeline_)
    vkDestroyPipeline(device, pipeline_, nullptr);
}

VkPipeline VulkanProgram::NativePipeline() const {
  return pipeline_;
}

VulkanProgram *VulkanProgram::Create(VulkanDevice *owner_device, const ProgramDesc &desc) {
  auto object = std::unique_ptr<VulkanProgram>(new VulkanProgram(owner_device, desc));
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
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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
}  // namespace grassland::rhi::backend::vulkan
