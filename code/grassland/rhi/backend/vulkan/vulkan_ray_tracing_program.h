#pragma once
#include <array>

#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanRayTracingProgram final : public RayTracingProgram, public VulkanProgramBase {
 public:
  VulkanRayTracingProgram(VulkanDevice *device, const RayTracingProgramDesc &desc);
  bool Finalize() override;
  ~VulkanRayTracingProgram() override;
  VkPipeline NativePipeline() const;
  const std::array<VkStridedDeviceAddressRegionKHR, 4> &ShaderTables() const;

 private:
  VkPipeline pipeline_{};
  ObjectPtr<VulkanAllocatedBuffer> table_;
  std::array<VkStridedDeviceAddressRegionKHR, 4> tables_{};
};
}  // namespace grassland::rhi::backend::vulkan
