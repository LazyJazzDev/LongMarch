#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanProgram final : public Program {
 public:
  VulkanProgram(VulkanDevice *device, const ProgramDesc &desc);
  ~VulkanProgram() override;
  bool Finalize() override;
  VkPipeline NativePipeline() const;

 private:
  VkPipeline pipeline_{};
  VkPipelineLayout layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
