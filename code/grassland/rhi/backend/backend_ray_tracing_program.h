#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class RayTracingProgram;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::RayTracingProgram)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {
class Device;

struct RayTracingProgramState;

class RayTracingProgram : public IRayTracingProgram {
 protected:
  void InitializeRayTracingProgram(Device *device, const RayTracingProgramDesc &desc);

 public:
  const RayTracingProgramDesc &GetDesc() const override;
  Device *Owner() const;

 protected:
  bool ValidateLayout() const;
  bool ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const;
  ShaderBindingTableDesc DefaultShaderBindingTableDesc() const;
  virtual RayTracingProgramState &StateRayTracingProgram() = 0;
  virtual const RayTracingProgramState &StateRayTracingProgram() const = 0;
};
}  // namespace grassland::rhi::backend
