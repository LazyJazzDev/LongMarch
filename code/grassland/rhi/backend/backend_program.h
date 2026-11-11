#pragma once
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_program.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendProgram)

namespace grassland::rhi::backend {
class IBackendProgram : public IProgram {
 public:
  const ProgramDesc &GetDesc() const override = 0;
  virtual IBackendDevice *Owner() const = 0;
};
}  // namespace grassland::rhi::backend
