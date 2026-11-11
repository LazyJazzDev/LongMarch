#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendRayTracingProgram)

namespace grassland::rhi::backend {
class IBackendRayTracingProgram : public IRayTracingProgram {
 public:
  const RayTracingProgramDesc &GetDesc() const override = 0;
  virtual IBackendDevice *Owner() const = 0;
};
}  // namespace grassland::rhi::backend
