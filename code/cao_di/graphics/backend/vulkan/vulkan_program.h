#pragma once
#include "cao_di/graphics/backend/vulkan/vulkan_core.h"
#include "cao_di/graphics/backend/vulkan/vulkan_util.h"

namespace CD::graphics::backend {
class VulkanShader : public Shader {
 public:
  VulkanShader(VulkanCore *core, const CompiledShaderBlob &shader_blob);
  ~VulkanShader() override = default;

  vulkan::ShaderModule *ShaderModule() const {
    return shader_module_.get();
  }

  const std::string &EntryPoint() const {
    return shader_module_->EntryPoint();
  }

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::ShaderModule> shader_module_;
};

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
  void BindShader(Shader *shader, ShaderType type) override;
  void Finalize() override;

  int NumInputBindings() const;
  const vulkan::PipelineSettings *PipelineSettings() const;

  vulkan::Pipeline *Pipeline() const {
    return pipeline_.get();
  }

 private:
  vulkan::PipelineSettings pipeline_settings_;
  std::unique_ptr<vulkan::Pipeline> pipeline_;
};

class VulkanComputeProgram : public ComputeProgram, public VulkanProgramBase {
 public:
  VulkanComputeProgram(VulkanCore *core, VulkanShader *compute_shader);
  ~VulkanComputeProgram() override;

  void AddResourceBinding(ResourceType type, int count) override;

  void Finalize() override;

  VkPipeline Pipeline() const {
    return pipeline_;
  }

 private:
  VulkanShader *compute_shader_;
  VkPipeline pipeline_;
};

class VulkanRayTracingProgram : public RayTracingProgram, public VulkanProgramBase {
 public:
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

  vulkan::RayTracingPipeline *Pipeline() const {
    return pipeline_.get();
  }

  vulkan::ShaderBindingTable *ShaderBindingTable() const {
    return shader_binding_table_.get();
  }

 private:
  vulkan::ShaderModule *raygen_shader_;
  std::vector<vulkan::ShaderModule *> miss_shaders_;
  std::vector<vulkan::HitGroup> hit_groups_;
  std::vector<vulkan::ShaderModule *> callable_shaders_;
  std::unique_ptr<vulkan::RayTracingPipeline> pipeline_;
  std::unique_ptr<vulkan::ShaderBindingTable> shader_binding_table_;
};

}  // namespace CD::graphics::backend
