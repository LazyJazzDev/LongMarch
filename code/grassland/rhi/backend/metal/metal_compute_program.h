#pragma once
#include "grassland/rhi/backend/metal/metal_program_base.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_compute_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalComputeProgram,
                        grassland::rhi::backend::metal::IMetalProgramBase,
                        "848b485d-2208-4d1d-86ae-cadee6182bfe")

namespace grassland::rhi::backend::metal {
class IMetalComputeProgram : public IMetalProgramBase {
 public:
  static rhi::IComputeProgram *Create(IMetalDevice *device, const ComputeProgramDesc &desc);
  virtual MTL::ComputePipelineState *NativePipeline() const = 0;
  virtual MTL::Size ThreadGroupSize() const = 0;
};

}  // namespace grassland::rhi::backend::metal
