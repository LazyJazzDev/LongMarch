#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"

namespace grassland::rhi::backend {
struct ComputeProgramState final {
  ComputeProgramState();
  ~ComputeProgramState();
  Device *device_{};
  ComputeProgramDesc desc_{};
  ObjectPtr<IShader> shader_{};
};
}  // namespace grassland::rhi::backend
