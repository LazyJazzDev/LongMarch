#pragma once
#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct RayTracingProgramState final {
  RayTracingProgramState();
  ~RayTracingProgramState();
  IBackendDevice *device_{};
  RayTracingProgramDesc desc_{};
  std::vector<ObjectPtr<IShader>> shaders_{};
};
}  // namespace grassland::rhi::backend
