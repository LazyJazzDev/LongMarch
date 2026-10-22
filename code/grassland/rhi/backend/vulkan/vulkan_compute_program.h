#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanComputeProgram final : public ComputeProgram, public VulkanProgramBase {
 public:
  static VulkanComputeProgram *Create(VulkanDevice *device, const ComputeProgramDesc &desc);
  ~VulkanComputeProgram() override;
  VkPipeline NativePipeline() const;

 private:
  VulkanComputeProgram(VulkanDevice *device, const ComputeProgramDesc &desc);
  VkPipeline pipeline_{};
};
}  // namespace grassland::rhi::backend::vulkan
