#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/metal/metal_program_base.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalComputeProgram)

namespace grassland::rhi::backend::metal {
class IMetalComputeProgram : public IMetalProgramBase {
 public:
  static rhi::IComputeProgram *Create(IMetalDevice *device, const ComputeProgramDesc &desc);
  virtual MTL::ComputePipelineState *NativePipeline() const = 0;
  virtual MTL::Size ThreadGroupSize() const = 0;
};

}  // namespace grassland::rhi::backend::metal
