#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanProgram, "d24a9c9f-6f6f-4489-a840-064c2a4309c1")

namespace grassland::rhi::backend::vulkan {
class IVulkanProgram : public IVulkanProgramBase {
 public:
  static rhi::IProgram *Create(IVulkanDevice *device, const ProgramDesc &desc);
  virtual VkPipeline NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
