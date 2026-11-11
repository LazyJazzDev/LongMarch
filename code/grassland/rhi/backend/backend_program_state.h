#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct ProgramState final {
  ProgramState();
  ~ProgramState();
  IBackendDevice *device_{};
  ProgramDesc desc_{};
  ObjectPtr<IShader> vertex_shader_{};
  ObjectPtr<IShader> fragment_shader_{};
};
}  // namespace grassland::rhi::backend
