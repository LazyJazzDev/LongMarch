#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_compute_program.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendComputeProgram)

namespace grassland::rhi::backend {
class IBackendComputeProgram : public IComputeProgram {
 public:
  const ComputeProgramDesc &GetDesc() const override = 0;
  virtual IBackendDevice *Owner() const = 0;
};
}  // namespace grassland::rhi::backend
