#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanComputeProgram final : public ComputeProgram {
 public:
  VulkanComputeProgram(VulkanDevice *device, const ComputeProgramDesc &desc);
  ~VulkanComputeProgram() override;
  bool Finalize() override;
  VkPipeline NativePipeline() const;
  VkPipelineLayout NativeLayout() const;

 private:
  std::vector<VkDescriptorSetLayout> set_layouts_;
  VkPipeline pipeline_{};
  VkPipelineLayout layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
