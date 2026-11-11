#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanRayTracingProgram)
#include <array>

#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanRayTracingProgram : public IBackendRayTracingProgram, public IVulkanProgramBase {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

 public:
  static IVulkanRayTracingProgram *Create(IVulkanDevice *device, const RayTracingProgramDesc &desc);
  virtual IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) = 0;
  virtual const IBackendShaderBindingTable *DefaultShaderBindingTable() const = 0;
  virtual VkPipeline NativePipeline() const = 0;
};

;
}  // namespace grassland::rhi::backend::vulkan
