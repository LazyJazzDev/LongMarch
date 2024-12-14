#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {
class VulkanShader : public Shader {
 public:
  VulkanShader(VulkanCore *core, const void *data, size_t size);
  ~VulkanShader() override = default;

  vulkan::ShaderModule *ShaderModule() const {
    return shader_module_.get();
  }

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::ShaderModule> shader_module_;
};

class VulkanProgram : public Program {
 public:
  VulkanProgram(VulkanCore *core,
                const std::vector<ImageFormat> &color_formats,
                ImageFormat depth_format);
  ~VulkanProgram() override;
  void AddInputAttribute(uint32_t binding,
                         InputType type,
                         uint32_t offset) override;
  void AddInputBinding(uint32_t stride, bool input_per_instance) override;
  void BindShader(Shader *shader, ShaderType type) override;
  void Finalize() override;

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::DescriptorSetLayout> descriptor_set_layout_;
  std::unique_ptr<vulkan::PipelineLayout> pipeline_layout_;
  vulkan::PipelineSettings pipeline_settings_;
  std::unique_ptr<vulkan::Pipeline> pipeline_;
};

}  // namespace grassland::graphics::backend
