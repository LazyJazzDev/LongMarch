#pragma once
#include "grassland/vulkan/device.h"
#include "grassland/vulkan/render_pass.h"
#include "grassland/vulkan/shader_module.h"

namespace CD::vulkan {
struct PipelineSettings {
  explicit PipelineSettings(const RenderPass *render_pass = nullptr,
                            const PipelineLayout *pipeline_layout = nullptr,
                            int subpass = 0);

  explicit PipelineSettings(const PipelineLayout *pipeline_layout = nullptr,
                            const std::vector<VkFormat> &color_attachment_formats = {},
                            VkFormat depth_attachment_format = VK_FORMAT_UNDEFINED);

  void PipelineSettingsCommon();

  void AddShaderStage(ShaderModule *shader_module, VkShaderStageFlagBits stage, const char *entry_point = "main");

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

  // Render pass
  const RenderPass *render_pass;

  // Pipeline layout
  const PipelineLayout *pipeline_layout;

  // Shader stages
  std::vector<VkPipelineShaderStageCreateInfo> shader_stage_create_infos;

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

class Pipeline {
 public:
  explicit Pipeline(const class Device *device, VkPipeline pipeline);

  virtual ~Pipeline();

  VkPipeline Handle() const {
    return pipeline_;
  }

  const class Device *Device() const {
    return device_;
  }

 private:
  const class Device *device_{};
  VkPipeline pipeline_{};
};

class RayTracingPipeline : public Pipeline {
 public:
  explicit RayTracingPipeline(const class Device *device,
                              VkPipeline pipeline,
                              size_t miss_shader_count,
                              size_t hit_group_count,
                              size_t callable_shader_count);

  ~RayTracingPipeline() = default;

  size_t MissShaderCount() const {
    return miss_shader_count_;
  }

  size_t HitGroupCount() const {
    return hit_group_count_;
  }

  size_t CallableShaderCount() const {
    return callable_shader_count_;
  }

 private:
  size_t miss_shader_count_{};
  size_t hit_group_count_{};
  size_t callable_shader_count_{};
};

}  // namespace CD::vulkan
