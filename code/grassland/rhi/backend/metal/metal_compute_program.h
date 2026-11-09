#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/metal/metal_program_base.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalComputeProgram final : public ComputeProgram, public MetalProgramBase {
 public:
  void AddRef() noexcept override {
    RetainReference();
  }

  void Release() noexcept override {
    if (ReleaseReference())
      delete this;
  }

  const ComputeProgramDesc &GetDesc() const override {
    return Description();
  }

  static MetalComputeProgram *Create(MetalDevice *device, const ComputeProgramDesc &desc);
  MTL::ComputePipelineState *NativePipeline() const;
  MTL::Size ThreadGroupSize() const;

 private:
  MetalComputeProgram(MetalDevice *device, const ComputeProgramDesc &desc);
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  MTL::Size threads_{};
};
}  // namespace grassland::rhi::backend::metal
