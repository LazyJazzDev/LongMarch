#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {
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

class VulkanRayTracingProgram : public RayTracingProgram, public VulkanProgramBase {
 public:
  VulkanRayTracingProgram(VulkanCore *core,
                          VulkanShader *raygen_shader,
                          VulkanShader *miss_shader,
                          VulkanShader *closest_hit_shader);
  ~VulkanRayTracingProgram() override = default;

  void AddResourceBinding(ResourceType type, int count) override;

  void Finalize() override;

  vulkan::Pipeline *Pipeline() const {
    return pipeline_.get();
  }

  vulkan::ShaderBindingTable *ShaderBindingTable() const {
    return shader_binding_table_.get();
  }

 private:
  VulkanShader *raygen_shader_;
  VulkanShader *miss_shader_;
  VulkanShader *closest_hit_shader_;
  std::unique_ptr<vulkan::Pipeline> pipeline_;
  std::unique_ptr<vulkan::ShaderBindingTable> shader_binding_table_;
};

}  // namespace grassland::graphics::backend
