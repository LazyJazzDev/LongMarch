#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendRayTracingProgram)
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {
struct RayTracingProgramState;

class IBackendRayTracingProgram : public IRayTracingProgram {
 protected:
  void InitializeRayTracingProgram(IBackendDevice *device, const RayTracingProgramDesc &desc);

 public:
  const RayTracingProgramDesc &GetDesc() const override = 0;
  const RayTracingProgramDesc &CommonGetDesc() const;
  IBackendDevice *Owner() const;

 protected:
  bool ValidateLayout() const;
  bool ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const;
  ShaderBindingTableDesc DefaultShaderBindingTableDesc() const;
  virtual RayTracingProgramState &StateRayTracingProgram() = 0;
  virtual const RayTracingProgramState &StateRayTracingProgram() const = 0;
};
}  // namespace grassland::rhi::backend
