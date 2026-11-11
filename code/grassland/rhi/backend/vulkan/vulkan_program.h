#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanProgram)

namespace grassland::rhi::backend::vulkan {
class IVulkanProgram : public IVulkanProgramBase {
 public:
  static rhi::IProgram *Create(IVulkanDevice *device, const ProgramDesc &desc);
  virtual VkPipeline NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
