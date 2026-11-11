#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_compute_program.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendComputeProgram, "f64ab67b-e062-4368-8605-dc1700c88e59")

namespace grassland::rhi::backend {
class IBackendComputeProgram : public IObject {
 public:
  virtual IBackendDevice *Owner() const = 0;
};
}  // namespace grassland::rhi::backend
