#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanProgram,
                        grassland::rhi::backend::vulkan::IVulkanProgramBase,
                        "d24a9c9f-6f6f-4489-a840-064c2a4309c1")

namespace grassland::rhi::backend::vulkan {
class IVulkanProgram : public IVulkanProgramBase {
 public:
  static rhi::IProgram *Create(IVulkanDevice *device, const ProgramDesc &desc);
  virtual VkPipeline NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
class VulkanProgramImpl final : public Object<rhi::IProgram, IVulkanProgram> {
 public:
  static rhi::IProgram *Create(IVulkanDevice *device, const ProgramDesc &desc);

  ~VulkanProgramImpl() override;

  VkPipeline NativePipeline() const;

  const ProgramDesc &GetDesc() const override;

  IVulkanDevice *Owner() const override;

  VkPipelineLayout NativeLayout() const override;

 private:
  VulkanProgramImpl(IVulkanDevice *device, const ProgramDesc &desc);

  VkPipeline pipeline_{};

  void InitializeProgram(IVulkanDevice *device, const ProgramDesc &desc);

  bool ValidateLayout() const;

  IVulkanDevice *device_{};

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
