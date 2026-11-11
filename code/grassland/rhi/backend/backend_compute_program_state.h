#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct ComputeProgramState final {
  ComputeProgramState();
  ~ComputeProgramState();
  IBackendDevice *device_{};
  ComputeProgramDesc desc_{};
  ObjectPtr<IShader> shader_{};
};
}  // namespace grassland::rhi::backend
