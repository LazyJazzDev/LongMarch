#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class VulkanRayTracingProgram;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::VulkanRayTracingProgram)
#include <array>

#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanRayTracingProgram final : public RayTracingProgram, public VulkanProgramBase {
 public:
  LONGMARCH_RHI_QUERY_MULTIPLE(VulkanRayTracingProgram, RayTracingProgram, VulkanProgramBase)

  static VulkanRayTracingProgram *Create(VulkanDevice *device, const RayTracingProgramDesc &desc);
  IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) override;
  const ShaderBindingTable *DefaultShaderBindingTable() const;
  ~VulkanRayTracingProgram() override;
  VkPipeline NativePipeline() const;

 private:
  VulkanRayTracingProgram(VulkanDevice *device, const RayTracingProgramDesc &desc);
  VkPipeline pipeline_{};
  ShaderBindingTable *BuildShaderBindingTable(const ShaderBindingTableDesc &desc);
  ObjectPtr<ShaderBindingTable> table_;
  std::array<std::vector<uint8_t>, 4> identifiers_;
  uint64_t handle_size_{}, stride_{}, alignment_{};
};
}  // namespace grassland::rhi::backend::vulkan
