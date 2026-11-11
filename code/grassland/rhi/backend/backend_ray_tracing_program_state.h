#pragma once
#include "grassland/rhi/backend/backend_ray_tracing_program.h"

namespace grassland::rhi::backend {
struct RayTracingProgramState final {
  RayTracingProgramState();
  ~RayTracingProgramState();
  Device *device_{};
  RayTracingProgramDesc desc_{};
  std::vector<ObjectPtr<IShader>> shaders_{};
};
}  // namespace grassland::rhi::backend
