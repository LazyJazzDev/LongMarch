#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalComputeProgram;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalComputeProgram)
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/metal/metal_program_base.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalComputeProgram final : public ComputeProgram, public MetalProgramBase {
 public:
  LONGMARCH_RHI_QUERY_MULTIPLE(MetalComputeProgram, ComputeProgram, MetalProgramBase)

  static MetalComputeProgram *Create(MetalDevice *device, const ComputeProgramDesc &desc);
  MTL::ComputePipelineState *NativePipeline() const;
  MTL::Size ThreadGroupSize() const;

 private:
  MetalComputeProgram(MetalDevice *device, const ComputeProgramDesc &desc);
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  MTL::Size threads_{};
};
}  // namespace grassland::rhi::backend::metal
