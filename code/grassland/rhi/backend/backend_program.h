#pragma once
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_program.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendProgram)

namespace grassland::rhi::backend {
struct ProgramState;

class IBackendProgram : public IProgram {
 protected:
  void InitializeProgram(IBackendDevice *device, const ProgramDesc &desc);
  void DestroyProgram();

 public:
  const ProgramDesc &GetDesc() const override = 0;
  const ProgramDesc &CommonGetDesc() const;
  IBackendDevice *Owner() const;

 protected:
  bool ValidateLayout() const;
  virtual ProgramState &StateProgram() = 0;
  virtual const ProgramState &StateProgram() const = 0;
};
}  // namespace grassland::rhi::backend
