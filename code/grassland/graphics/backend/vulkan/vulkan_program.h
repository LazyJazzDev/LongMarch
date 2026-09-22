#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_shader.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanProgramBase {
 public:
  VulkanProgramBase(VulkanCore *core);
  virtual ~VulkanProgramBase() = default;

  vulkan::PipelineLayout *PipelineLayout() const {
    return pipeline_layout_.get();
  }

  vulkan::DescriptorSetLayout *DescriptorSetLayout(int index) const {
    return descriptor_set_layouts_[index].get();
  }

  VulkanCore *Core() const {
    return core_;
  }

 protected:
  void AddResourceBindingImpl(ResourceType type, int count);

  void FinalizePipelineLayout();
  VulkanCore *core_;
  std::vector<std::unique_ptr<vulkan::DescriptorSetLayout>> descriptor_set_layouts_;
  std::unique_ptr<vulkan::PipelineLayout> pipeline_layout_;
};

class VulkanProgram : public Program, public VulkanProgramBase {
 public:
  VulkanProgram(VulkanCore *core, const std::vector<ImageFormat> &color_formats, ImageFormat depth_format);
  VulkanProgram(VulkanCore *core,
                VulkanShader *raygen_shader,
                VulkanShader *miss_shader,
                VulkanShader *closest_hit_shader);
  ~VulkanProgram() override;
  void AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) override;
  void AddInputBinding(uint32_t stride, bool input_per_instance) override;
  void AddResourceBinding(ResourceType type, int count) override;
  void SetCullMode(CullMode mode) override;
  void SetBlendState(int target_id, const BlendState &state) override;
  using Program::BindShader;
  void BindShader(Shader *shader, ShaderType type) override;
  void Finalize() override;

  int NumInputBindings() const;
  const vulkan::PipelineSettings *PipelineSettings() const;

  vulkan::Pipeline *Pipeline() const {
    return pipeline_.get();
  }

 private:
  std::map<ShaderType, Shader *> shader_stages_;
  vulkan::PipelineSettings pipeline_settings_;
  std::unique_ptr<vulkan::Pipeline> pipeline_;
};

class VulkanComputeProgram : public ComputeProgram, public VulkanProgramBase {
 public:
  using ComputeProgram::BindShader;

  void BindShader(Shader *shader) override {
    compute_shader_ = shader;
  }

  VulkanComputeProgram(VulkanCore *core, Shader *compute_shader);
  ~VulkanComputeProgram() override;

  void AddResourceBinding(ResourceType type, int count) override;

  void Finalize() override;

  VkPipeline Pipeline() const {
    return pipeline_;
  }

 private:
  Shader *compute_shader_;
  VkPipeline pipeline_{VK_NULL_HANDLE};
};

class VulkanRayTracingProgram : public RayTracingProgram, public VulkanProgramBase {
 public:
  using RayTracingProgram::AddCallableShader;
  using RayTracingProgram::AddHitGroup;
  using RayTracingProgram::AddMissShader;
  using RayTracingProgram::AddRayGenShader;
  VulkanRayTracingProgram(VulkanCore *core);
  VulkanRayTracingProgram(VulkanCore *core,
                          VulkanShader *raygen_shader,
                          VulkanShader *miss_shader,
                          VulkanShader *closest_hit_shader);
  ~VulkanRayTracingProgram() override = default;

  void AddResourceBinding(ResourceType type, int count) override;

  void AddRayGenShader(Shader *ray_gen_shader) override;
  void AddMissShader(Shader *miss_shader) override;
  void AddHitGroup(HitGroup hit_group) override;
  void AddCallableShader(Shader *callable_shader) override;

  void Finalize(const std::vector<int32_t> &miss_shader_indices,
                const std::vector<int32_t> &hit_group_indices,
                const std::vector<int32_t> &callable_shader_indices) override;
  void Finalize() override;

  vulkan::RayTracingPipeline *Pipeline() const {
    return pipeline_.get();
  }

  vulkan::ShaderBindingTable *ShaderBindingTable() const {
    return shader_binding_table_.get();
  }

 private:
  void ResolveShaders();
  Shader *source_raygen_ = nullptr;
  std::vector<Shader *> source_miss_, source_callable_;
  std::vector<HitGroup> source_hit_groups_;
  vulkan::ShaderModule *raygen_shader_ = nullptr;
  std::vector<vulkan::ShaderModule *> miss_shaders_;
  std::vector<vulkan::HitGroup> hit_groups_;
  std::vector<vulkan::ShaderModule *> callable_shaders_;
  std::unique_ptr<vulkan::RayTracingPipeline> pipeline_;
  std::unique_ptr<vulkan::ShaderBindingTable> shader_binding_table_;
};

}  // namespace grassland::graphics::backend
