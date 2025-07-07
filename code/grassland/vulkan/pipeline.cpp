#include "grassland/vulkan/pipeline.h"

#include <utility>

namespace grassland::vulkan {

PipelineSettings::PipelineSettings(const RenderPass *render_pass, const PipelineLayout *pipeline_layout, int subpass)
    : render_pass(render_pass), pipeline_layout(pipeline_layout), subpass(subpass) {
  PipelineSettingsCommon();

  auto &subpass_settings = render_pass->SubpassSettings()[subpass];

  if (render_pass) {
    if (subpass_settings.DepthAttachmentReference().has_value()) {
      depth_stencil_state_create_info = VkPipelineDepthStencilStateCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
          nullptr,
          0,
          VK_TRUE,
          VK_TRUE,
          VK_COMPARE_OP_LESS,
          VK_FALSE,
          VK_FALSE,
          VkStencilOpState{},
          VkStencilOpState{},
          0.0f,
          1.0f,
      };
    }

    if (!subpass_settings.ColorAttachmentReferences().empty()) {
      pipeline_color_blend_attachment_states.resize(subpass_settings.ColorAttachmentReferences().size());
      for (size_t i = 0; i < subpass_settings.ColorAttachmentReferences().size(); i++) {
        pipeline_color_blend_attachment_states[i].blendEnable = VK_FALSE;
        pipeline_color_blend_attachment_states[i].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        pipeline_color_blend_attachment_states[i].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        pipeline_color_blend_attachment_states[i].colorBlendOp = VK_BLEND_OP_ADD;
        pipeline_color_blend_attachment_states[i].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        pipeline_color_blend_attachment_states[i].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        pipeline_color_blend_attachment_states[i].alphaBlendOp = VK_BLEND_OP_ADD;
        pipeline_color_blend_attachment_states[i].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      }
    }

    if (!subpass_settings.ResolveAttachmentReferences().empty()) {
      multisample_state_create_info.alphaToCoverageEnable = VK_TRUE;
      multisample_state_create_info.alphaToOneEnable = VK_TRUE;
    }
  }
}

PipelineSettings::PipelineSettings(const PipelineLayout *pipeline_layout,
                                   const std::vector<VkFormat> &color_attachment_formats,
                                   VkFormat depth_attachment_format)
    : render_pass(nullptr),
      pipeline_layout(pipeline_layout),
      color_attachment_formats(color_attachment_formats),
      depth_attachment_format(depth_attachment_format),
      subpass(0) {
  PipelineSettingsCommon();
  if (depth_attachment_format != VK_FORMAT_UNDEFINED) {
    depth_stencil_state_create_info = VkPipelineDepthStencilStateCreateInfo{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        nullptr,
        0,
        VK_TRUE,
        VK_TRUE,
        VK_COMPARE_OP_LESS,
        VK_FALSE,
        VK_FALSE,
        VkStencilOpState{},
        VkStencilOpState{},
        0.0f,
        1.0f,
    };
  }

  if (!color_attachment_formats.empty()) {
    pipeline_color_blend_attachment_states.resize(color_attachment_formats.size());
    for (size_t i = 0; i < color_attachment_formats.size(); i++) {
      pipeline_color_blend_attachment_states[i].blendEnable = VK_FALSE;
      pipeline_color_blend_attachment_states[i].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      pipeline_color_blend_attachment_states[i].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      pipeline_color_blend_attachment_states[i].colorBlendOp = VK_BLEND_OP_ADD;
      pipeline_color_blend_attachment_states[i].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      pipeline_color_blend_attachment_states[i].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      pipeline_color_blend_attachment_states[i].alphaBlendOp = VK_BLEND_OP_ADD;
      pipeline_color_blend_attachment_states[i].colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
  }
}

void PipelineSettings::PipelineSettingsCommon() {
  input_assembly_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_state_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly_state_create_info.primitiveRestartEnable = VK_FALSE;

  multisample_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample_state_create_info.sampleShadingEnable = VK_FALSE;
  multisample_state_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisample_state_create_info.minSampleShading = 1.0f;
  multisample_state_create_info.pSampleMask = nullptr;
  multisample_state_create_info.alphaToCoverageEnable = VK_FALSE;
  multisample_state_create_info.alphaToOneEnable = VK_FALSE;

  rasterization_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization_state_create_info.depthClampEnable = VK_FALSE;
  rasterization_state_create_info.rasterizerDiscardEnable = VK_FALSE;
  rasterization_state_create_info.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization_state_create_info.lineWidth = 1.0f;
  rasterization_state_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterization_state_create_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization_state_create_info.depthBiasEnable = VK_FALSE;
}

void PipelineSettings::AddInputBinding(uint32_t binding, uint32_t stride, VkVertexInputRate input_rate) {
  vertex_input_binding_descriptions.push_back(VkVertexInputBindingDescription{
      binding,
      stride,
      input_rate,
  });
}

void PipelineSettings::AddInputAttribute(uint32_t binding, uint32_t location, VkFormat format, uint32_t offset) {
  vertex_input_attribute_descriptions.push_back(VkVertexInputAttributeDescription{
      location,
      binding,
      format,
      offset,
  });
}

void PipelineSettings::AddShaderStage(ShaderModule *shader_module,
                                      VkShaderStageFlagBits stage,
                                      const char *entry_point) {
  shader_stage_create_infos.push_back(VkPipelineShaderStageCreateInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      nullptr,
      0,
      stage,
      shader_module->Handle(),
      entry_point,
      nullptr,
  });
}

void PipelineSettings::SetPrimitiveTopology(VkPrimitiveTopology topology) {
  input_assembly_state_create_info.topology = topology;
}

void PipelineSettings::SetMultiSampleState(VkSampleCountFlagBits sample_count) {
  multisample_state_create_info.rasterizationSamples = sample_count;
}

void PipelineSettings::SetCullMode(VkCullModeFlags cull_mode) {
  rasterization_state_create_info.cullMode = cull_mode;
}

void PipelineSettings::SetPolygonMode(VkPolygonMode polygon_mode) {
  rasterization_state_create_info.polygonMode = polygon_mode;
}

void PipelineSettings::SetSubpass(int subpass) {
  this->subpass = subpass;
}

void PipelineSettings::SetBlendState(int color_attachment_index,
                                     VkPipelineColorBlendAttachmentState blend_attachment_state) {
  pipeline_color_blend_attachment_states[color_attachment_index] = blend_attachment_state;
}

void PipelineSettings::SetTessellationState(uint32_t patch_control_points) {
  tessellation_state_create_info = VkPipelineTessellationStateCreateInfo{
      VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
      nullptr,
      0,
      patch_control_points,
  };
}

void PipelineSettings::EnableDynamicPrimitiveTopology() {
  dynamic_primitive_topology = true;
}

Pipeline::Pipeline(const class vulkan::Device *device, VkPipeline pipeline) : device_(device), pipeline_(pipeline) {
}

Pipeline::~Pipeline() {
  vkDestroyPipeline(device_->Handle(), pipeline_, nullptr);
}

RayTracingPipeline::RayTracingPipeline(const class vulkan::Device *device,
                                       VkPipeline pipeline,
                                       size_t miss_shader_count,
                                       size_t hit_group_count,
                                       size_t callable_shader_count)
    : Pipeline(device, pipeline),
      miss_shader_count_(miss_shader_count),
      hit_group_count_(hit_group_count),
      callable_shader_count_(callable_shader_count) {
}

}  // namespace grassland::vulkan
