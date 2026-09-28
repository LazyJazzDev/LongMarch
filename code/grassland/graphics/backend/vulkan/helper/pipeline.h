#pragma once
#include "grassland/graphics/backend/vulkan/helper/native_types.h"
#include "grassland/graphics/backend/vulkan/helper/shader_module.h"

namespace grassland::graphics::backend::vulkan {
struct PipelineSettings {
  explicit PipelineSettings(VkPipelineLayout pipeline_layout = VK_NULL_HANDLE,
                            const std::vector<VkFormat> &color_attachment_formats = {},
                            VkFormat depth_attachment_format = VK_FORMAT_UNDEFINED);

  void PipelineSettingsCommon();

  void AddShaderStage(VkShaderModule shader_module, const std::string &entry_point, VkShaderStageFlagBits stage);

  void AddInputBinding(uint32_t binding, uint32_t stride, VkVertexInputRate input_rate = VK_VERTEX_INPUT_RATE_VERTEX);

  void AddInputAttribute(uint32_t binding, uint32_t location, VkFormat format, uint32_t offset);

  void SetPrimitiveTopology(VkPrimitiveTopology topology);

  void SetMultiSampleState(VkSampleCountFlagBits sample_count);

  void SetCullMode(VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT);

  void SetPolygonMode(VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL);

  void SetSubpass(int subpass);

  void SetBlendState(int color_attachment_index,
                     VkPipelineColorBlendAttachmentState blend_attachment_state = {
                         VK_FALSE,
                         VK_BLEND_FACTOR_ONE,
                         VK_BLEND_FACTOR_ZERO,
                         VK_BLEND_OP_ADD,
                         VK_BLEND_FACTOR_ONE,
                         VK_BLEND_FACTOR_ZERO,
                         VK_BLEND_OP_ADD,
                         VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                             VK_COLOR_COMPONENT_A_BIT,
                     });

  void SetTessellationState(uint32_t patch_control_points);

  void EnableDynamicPrimitiveTopology();

  // Pipeline layout
  VkPipelineLayout pipeline_layout;

  // Shader stages
  std::vector<VkPipelineShaderStageCreateInfo> shader_stage_create_infos;
  std::vector<std::unique_ptr<char[]>> shader_entry_points_storage;

  // Vertex input
  std::vector<VkVertexInputBindingDescription> vertex_input_binding_descriptions;
  std::vector<VkVertexInputAttributeDescription> vertex_input_attribute_descriptions;

  // Blend state
  std::vector<VkPipelineColorBlendAttachmentState> pipeline_color_blend_attachment_states;

  // Primitive topology
  VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info{};
  bool dynamic_primitive_topology{false};

  // Depth stencil state
  std::optional<VkPipelineDepthStencilStateCreateInfo> depth_stencil_state_create_info;

  // Multisample state
  VkPipelineMultisampleStateCreateInfo multisample_state_create_info{};

  // Rasterization state
  VkPipelineRasterizationStateCreateInfo rasterization_state_create_info{};

  std::optional<VkPipelineTessellationStateCreateInfo> tessellation_state_create_info;

  int subpass{};

  // Dynamic rendering state
  std::vector<VkFormat> color_attachment_formats{};
  VkFormat depth_attachment_format{VK_FORMAT_UNDEFINED};
};

}  // namespace grassland::graphics::backend::vulkan
