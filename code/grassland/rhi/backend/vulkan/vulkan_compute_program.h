#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanComputeProgram final : public ComputeProgram, public VulkanProgramBase {
 public:
  VulkanComputeProgram(VulkanDevice *device, const ComputeProgramDesc &desc);
  ~VulkanComputeProgram() override;
  bool Finalize() override;
  VkPipeline NativePipeline() const;

 private:
  VkPipeline pipeline_{};
};
}  // namespace grassland::rhi::backend::vulkan
