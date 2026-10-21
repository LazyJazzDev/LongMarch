#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanProgram final : public Program, public VulkanProgramBase {
 public:
  VulkanProgram(VulkanDevice *device, const ProgramDesc &desc);
  ~VulkanProgram() override;
  bool Finalize() override;
  VkPipeline NativePipeline() const;

 private:
  VkPipeline pipeline_{};
};
}  // namespace grassland::rhi::backend::vulkan
