#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

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
