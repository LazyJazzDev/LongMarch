#include "grassland/rhi/backend/vulkan/vulkan_program.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanProgram::VulkanProgram(VulkanDevice *device, const ProgramDesc &desc) : Program(device, desc) {
}

VulkanProgram::~VulkanProgram() {
  auto device = static_cast<VulkanDevice *>(device_)->NativeDevice();
  if (pipeline_)
    vkDestroyPipeline(device, pipeline_, nullptr);
  if (layout_)
    vkDestroyPipelineLayout(device, layout_, nullptr);
  for (auto layout : set_layouts_)
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
}

VkPipeline VulkanProgram::NativePipeline() const {
  return pipeline_;
}

VkPipelineLayout VulkanProgram::NativeLayout() const {
  return layout_;
}

bool VulkanProgram::Finalize() {
  if (finalized_)
    return true;
  auto owner = static_cast<VulkanDevice *>(device_);
  auto device = owner->NativeDevice();
  VkShaderModule modules[2]{};
  auto cleanup = [&]() {
    for (auto module : modules)
      if (module)
        vkDestroyShaderModule(device, module, nullptr);
  };
  try {
    if (!ValidateLayout() || !ValidatePipelineBindings(owner->NativePhysicalDevice(), desc_.pipeline_layout)) {
      ReportError(-1, "Invalid graphics program layout.");
      return false;
    }
    std::vector<uint8_t> code[2];
    auto prefix = ShaderMacroHeader(desc_, BACKEND_API_VULKAN);
    if (!CompileSlang(desc_.vertex_shader->GetDesc(), prefix, BACKEND_API_VULKAN, code[0], &desc_.pipeline_layout))
      return false;
    if (!CompileSlang(desc_.fragment_shader->GetDesc(), prefix, BACKEND_API_VULKAN, code[1], &desc_.pipeline_layout))
      return false;
    for (int i = 0; i < 2; ++i) {
      VkShaderModuleCreateInfo create{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      create.codeSize = code[i].size();
      create.pCode = reinterpret_cast<const uint32_t *>(code[i].data());
      if (!Check(vkCreateShaderModule(device, &create, nullptr, &modules[i]), "Create shader module")) {
        cleanup();
        return false;
      }
    }
    if (!layout_) {
      for (auto layout : set_layouts_)
        vkDestroyDescriptorSetLayout(device, layout, nullptr);
      set_layouts_.clear();
      set_layouts_.reserve(desc_.pipeline_layout.groups.size());
      for (const auto &group : desc_.pipeline_layout.groups) {
        auto layout = CreateBindingLayout(device, group);
        if (!layout) {
          cleanup();
          return false;
        }
        set_layouts_.push_back(layout);
      }
      VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      create.setLayoutCount = uint32_t(set_layouts_.size());
      create.pSetLayouts = set_layouts_.data();
      if (!Check(vkCreatePipelineLayout(device, &create, nullptr, &layout_), "Create pipeline layout")) {
        cleanup();
        return false;
      }
    }
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (int i = 0; i < 2; ++i) {
      stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[i].module = modules[i];
    }
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].pName = desc_.vertex_shader->GetDesc().entry_point.c_str();
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].pName = desc_.fragment_shader->GetDesc().entry_point.c_str();
    std::vector<VkVertexInputBindingDescription> bindings;
    for (uint32_t i = 0; i < desc_.vertex_bindings.size(); ++i) {
      const auto &b = desc_.vertex_bindings[i];
      bindings.push_back({i, b.stride, b.per_instance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
    }
    const VkFormat formats[] = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT,
                                VK_FORMAT_R32G32B32A32_SFLOAT};
    std::vector<VkVertexInputAttributeDescription> attributes;
    for (const auto &a : desc_.vertex_attributes)
      attributes.push_back({a.location, a.binding, formats[a.format], a.offset});
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
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(desc_.color_formats.size(), attachment);
    blend.attachmentCount = uint32_t(blend_attachments.size());
    blend.pAttachments = blend_attachments.data();
    const auto &ds = desc_.depth_stencil;
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
    create.layout = layout_;
    std::vector<VkFormat> color_formats;
    for (auto format : desc_.color_formats)
      color_formats.push_back(ColorFormat(format));
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    if (owner->BeginRenderingCommand()) {
      rendering.colorAttachmentCount = uint32_t(color_formats.size());
      rendering.pColorAttachmentFormats = color_formats.data();
      rendering.depthAttachmentFormat = ColorFormat(ds.format);
      rendering.stencilAttachmentFormat = HasStencil(ds.format) ? ColorFormat(ds.format) : VK_FORMAT_UNDEFINED;
      create.pNext = &rendering;
    } else {
      create.renderPass = owner->RenderTargets().RenderPass(desc_.color_formats, ds.format);
      if (!create.renderPass) {
        cleanup();
        return false;
      }
    }
    auto result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create, nullptr, &pipeline_);
    cleanup();
    finalized_ = Check(result, "Create graphics pipeline");
    if (!finalized_ && pipeline_) {
      vkDestroyPipeline(device, pipeline_, nullptr);
      pipeline_ = {};
    }
    return finalized_;
  } catch (const std::bad_alloc &) {
    cleanup();
    ReportError(-1, "Program allocation failed.");
    return false;
  }
}
}  // namespace grassland::rhi::backend::vulkan
