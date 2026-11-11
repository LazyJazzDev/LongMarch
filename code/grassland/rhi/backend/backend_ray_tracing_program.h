#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendRayTracingProgram,
                        grassland::rhi::IObject,
                        "e0b9733f-8a37-4702-800c-3022865d522c")

namespace grassland::rhi::backend {
class IBackendRayTracingProgram : public IObject {
 public:
  virtual IBackendDevice *Owner() const = 0;
};
}  // namespace grassland::rhi::backend
