#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {
class Device;

class RayTracingProgram : public Object<IRayTracingProgram> {
 public:
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
