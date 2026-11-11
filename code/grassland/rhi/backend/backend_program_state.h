#pragma once
#include "grassland/rhi/backend/backend_program.h"

namespace grassland::rhi::backend {
struct ProgramState final {
  ProgramState();
  ~ProgramState();
  Device *device_{};
  ProgramDesc desc_{};
  ObjectPtr<IShader> vertex_shader_{};
  ObjectPtr<IShader> fragment_shader_{};
};
}  // namespace grassland::rhi::backend
