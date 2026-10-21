#pragma once
#include <array>

#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanRayTracingProgram final : public RayTracingProgram, public VulkanProgramBase {
 public:
  VulkanRayTracingProgram(VulkanDevice *device, const RayTracingProgramDesc &desc);
  bool Finalize() override;
  IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) override;
  const ShaderBindingTable *DefaultShaderBindingTable() const;
  ~VulkanRayTracingProgram() override;
  VkPipeline NativePipeline() const;

 private:
  VkPipeline pipeline_{};
  ShaderBindingTable *BuildShaderBindingTable(const ShaderBindingTableDesc &desc);
  ObjectPtr<ShaderBindingTable> table_;
  std::array<std::vector<uint8_t>, 4> identifiers_;
  uint64_t handle_size_{}, stride_{}, alignment_{};
};
}  // namespace grassland::rhi::backend::vulkan
