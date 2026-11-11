#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanComputeProgramImpl final : public Object<IVulkanComputeProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IComputeProgram,
                            IVulkanComputeProgram,
                            IBackendComputeProgram,
                            IComputeProgram,
                            IVulkanProgramBase,
                            IObject)

  static IVulkanComputeProgram *Create(IVulkanDevice *device, const ComputeProgramDesc &desc);

  ~VulkanComputeProgramImpl() override;

  VkPipeline NativePipeline() const;

  const ComputeProgramDesc &GetDesc() const override;

  IBackendDevice *Owner() const override;

  VkPipelineLayout NativeLayout() const override;

 private:
  VulkanComputeProgramImpl(IVulkanDevice *device, const ComputeProgramDesc &desc);

  VkPipeline pipeline_{};

  void InitializeComputeProgram(IBackendDevice *device, const ComputeProgramDesc &desc);

  bool ValidateLayout() const;

  IBackendDevice *device_{};

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
