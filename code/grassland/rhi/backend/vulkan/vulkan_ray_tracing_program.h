#pragma once
#include <array>

#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanRayTracingProgram)

namespace grassland::rhi::backend::vulkan {
class IVulkanRayTracingProgram : public IVulkanProgramBase {
 public:
  static rhi::IRayTracingProgram *Create(IVulkanDevice *device, const RayTracingProgramDesc &desc);
  virtual const IBackendShaderBindingTable *DefaultShaderBindingTable() const = 0;
  virtual VkPipeline NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
