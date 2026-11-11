#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class VulkanProgram;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::VulkanProgram)
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanProgram final : public Program, public VulkanProgramBase {
 public:
  LONGMARCH_RHI_QUERY_MULTIPLE(VulkanProgram, Program, VulkanProgramBase)

  static VulkanProgram *Create(VulkanDevice *device, const ProgramDesc &desc);
  ~VulkanProgram() override;
  VkPipeline NativePipeline() const;

 private:
  VulkanProgram(VulkanDevice *device, const ProgramDesc &desc);
  VkPipeline pipeline_{};
};
}  // namespace grassland::rhi::backend::vulkan
