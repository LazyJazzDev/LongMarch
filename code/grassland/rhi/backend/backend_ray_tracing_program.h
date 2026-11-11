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

class RayTracingProgram : public Object<IRayTracingProgram> {
 public:
  LONGMARCH_RHI_QUERY(RayTracingProgram, Object<IRayTracingProgram>)

  RayTracingProgram(Device *device, const RayTracingProgramDesc &desc);
  const RayTracingProgramDesc &GetDesc() const override;
  Device *Owner() const;

 protected:
  bool ValidateLayout() const;
  bool ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const;
  ShaderBindingTableDesc DefaultShaderBindingTableDesc() const;
  Device *device_;
  RayTracingProgramDesc desc_;

 private:
  std::vector<ObjectPtr<IShader>> shaders_;
};
}  // namespace grassland::rhi::backend
