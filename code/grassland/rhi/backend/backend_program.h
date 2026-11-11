#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Program;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Program)
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
class Device;

struct ProgramState;

class Program : public IProgram {
 protected:
  void InitializeProgram(Device *device, const ProgramDesc &desc);
  void DestroyProgram();

 public:
  const ProgramDesc &GetDesc() const override;
  Device *Owner() const;

 protected:
  bool ValidateLayout() const;
  virtual ProgramState &StateProgram() = 0;
  virtual const ProgramState &StateProgram() const = 0;
};
}  // namespace grassland::rhi::backend
