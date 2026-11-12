#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_compute_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanComputeProgram,
                        grassland::rhi::backend::vulkan::IVulkanProgramBase,
                        "3ce12e7b-3db2-461e-b7d0-9ec0f172ebd1")

namespace grassland::rhi::backend::vulkan {
class IVulkanComputeProgram : public IVulkanProgramBase {
 public:
  static rhi::IComputeProgram *Create(IVulkanDevice *device, const ComputeProgramDesc &desc);
  virtual VkPipeline NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
class VulkanComputeProgramImpl final : public Object<rhi::IComputeProgram, IVulkanComputeProgram> {
 public:
  static rhi::IComputeProgram *Create(IVulkanDevice *device, const ComputeProgramDesc &desc);

  ~VulkanComputeProgramImpl() override;

  VkPipeline NativePipeline() const;

  const ComputeProgramDesc &GetDesc() const override;

  IVulkanDevice *Owner() const override;

  VkPipelineLayout NativeLayout() const override;

 private:
  VulkanComputeProgramImpl(IVulkanDevice *device, const ComputeProgramDesc &desc);

  VkPipeline pipeline_{};

  void InitializeComputeProgram(IVulkanDevice *device, const ComputeProgramDesc &desc);

  bool ValidateLayout() const;

  IVulkanDevice *device_{};

  ComputeProgramDesc desc_{};

  ObjectPtr<IShader> shader_{};

  void InitializeVulkanProgramBase(VkDevice device);

  void DestroyVulkanProgramBase();

  bool CreateLayout(const PipelineLayoutDesc &desc);

  VkDevice LayoutDevice() const;

  VkDevice layout_device_{};
  std::vector<VkDescriptorSetLayout> set_layouts_{};
  VkPipelineLayout layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
