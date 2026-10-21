#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/metal/metal_program_base.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalComputeProgram final : public ComputeProgram, public MetalProgramBase {
 public:
  MetalComputeProgram(MetalDevice *device, const ComputeProgramDesc &desc);
  bool Finalize() override;
  MTL::ComputePipelineState *NativePipeline() const;
  MTL::Size ThreadGroupSize() const;

 private:
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  MTL::Size threads_{};
};
}  // namespace grassland::rhi::backend::metal
