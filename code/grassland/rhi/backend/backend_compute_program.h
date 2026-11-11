#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class ComputeProgram;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::ComputeProgram)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_compute_program.h"

namespace grassland::rhi::backend {
class Device;

struct ComputeProgramState;

class ComputeProgram : public IComputeProgram {
 protected:
  void InitializeComputeProgram(Device *device, const ComputeProgramDesc &desc);

 public:
  const ComputeProgramDesc &GetDesc() const override;
  Device *Owner() const;

 protected:
  bool ValidateLayout() const;
  virtual ComputeProgramState &StateComputeProgram() = 0;
  virtual const ComputeProgramState &StateComputeProgram() const = 0;
};
}  // namespace grassland::rhi::backend
