#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendComputeProgram)
#include "grassland/rhi/rhi_compute_program.h"

namespace grassland::rhi::backend {
struct ComputeProgramState;

class IBackendComputeProgram : public IComputeProgram {
 protected:
  void InitializeComputeProgram(IBackendDevice *device, const ComputeProgramDesc &desc);

 public:
  const ComputeProgramDesc &GetDesc() const override = 0;
  const ComputeProgramDesc &CommonGetDesc() const;
  IBackendDevice *Owner() const;

 protected:
  bool ValidateLayout() const;
  virtual ComputeProgramState &StateComputeProgram() = 0;
  virtual const ComputeProgramState &StateComputeProgram() const = 0;
};
}  // namespace grassland::rhi::backend
