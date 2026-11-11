#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanProgramImpl final : public Object<rhi::IProgram, IVulkanProgram, IBackendProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IProgram, IVulkanProgram, IBackendProgram, IProgram, IVulkanProgramBase)

  static rhi::IProgram *Create(IVulkanDevice *device, const ProgramDesc &desc);

  ~VulkanProgramImpl() override;

  VkPipeline NativePipeline() const;

  const ProgramDesc &GetDesc() const override;

  IBackendDevice *Owner() const override;

  VkPipelineLayout NativeLayout() const override;

 private:
  VulkanProgramImpl(IVulkanDevice *device, const ProgramDesc &desc);

  VkPipeline pipeline_{};

  void InitializeProgram(IBackendDevice *device, const ProgramDesc &desc);

  bool ValidateLayout() const;

  IBackendDevice *device_{};

  ProgramDesc desc_{};

  ObjectPtr<IShader> vertex_shader_{};

  ObjectPtr<IShader> fragment_shader_{};

  void InitializeVulkanProgramBase(VkDevice device);

  void DestroyVulkanProgramBase();

  bool CreateLayout(const PipelineLayoutDesc &desc);

  VkDevice LayoutDevice() const;

  VkDevice layout_device_{};
  std::vector<VkDescriptorSetLayout> set_layouts_{};
  VkPipelineLayout layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
