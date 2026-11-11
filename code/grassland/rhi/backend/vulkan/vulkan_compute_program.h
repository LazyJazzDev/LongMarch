#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanComputeProgram)

namespace grassland::rhi::backend::vulkan {
class IVulkanComputeProgram : public IVulkanProgramBase {
 public:
  static rhi::IComputeProgram *Create(IVulkanDevice *device, const ComputeProgramDesc &desc);
  virtual VkPipeline NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
