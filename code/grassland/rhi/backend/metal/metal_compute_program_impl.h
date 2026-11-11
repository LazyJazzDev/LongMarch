#pragma once
#include "grassland/rhi/backend/backend_compute_program_state.h"
#include "grassland/rhi/backend/metal/metal_compute_program.h"
#include "grassland/rhi/backend/metal/metal_program_base_state.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalComputeProgramImpl final : public Object<IMetalComputeProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IComputeProgram,
                            IMetalComputeProgram,
                            IBackendComputeProgram,
                            IComputeProgram,
                            IMetalProgramBase,
                            IObject)
  static IMetalComputeProgram *Create(IMetalDevice *device, const ComputeProgramDesc &desc);
  MTL::ComputePipelineState *NativePipeline() const;
  MTL::Size ThreadGroupSize() const;

 private:
  MetalComputeProgramImpl(IMetalDevice *device, const ComputeProgramDesc &desc);
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  MTL::Size threads_{};

 public:
  ~MetalComputeProgramImpl() override {
  }

 private:
  ComputeProgramState state_ComputeProgram_;

  ComputeProgramState &StateComputeProgram() override {
    return state_ComputeProgram_;
  }

  const ComputeProgramState &StateComputeProgram() const override {
    return state_ComputeProgram_;
  }

 private:
  MetalProgramBaseState state_MetalProgramBase_;

  MetalProgramBaseState &StateMetalProgramBase() override {
    return state_MetalProgramBase_;
  }

  const MetalProgramBaseState &StateMetalProgramBase() const override {
    return state_MetalProgramBase_;
  }

 public:
  const ComputeProgramDesc &GetDesc() const override {
    return IBackendComputeProgram::CommonGetDesc();
  }
};
}  // namespace grassland::rhi::backend::metal
