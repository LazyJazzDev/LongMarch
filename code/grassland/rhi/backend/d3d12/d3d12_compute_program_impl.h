#pragma once
#include "grassland/rhi/backend/backend_compute_program_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_compute_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12ComputeProgramImpl final : public Object<ID3D12ComputeProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IComputeProgram,
                            ID3D12ComputeProgram,
                            IBackendComputeProgram,
                            IComputeProgram,
                            ID3D12ProgramBase,
                            IObject)
  static ID3D12ComputeProgram *Create(ID3D12Device *device, const ComputeProgramDesc &desc);
  ID3D12PipelineState *NativePipeline() const;

 private:
  D3D12ComputeProgramImpl(ID3D12Device *device, const ComputeProgramDesc &desc);
  ComPtr<ID3D12PipelineState> pipeline_;

 public:
  ~D3D12ComputeProgramImpl() override {
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
  D3D12ProgramBaseState state_D3D12ProgramBase_;

  D3D12ProgramBaseState &StateD3D12ProgramBase() override {
    return state_D3D12ProgramBase_;
  }

  const D3D12ProgramBaseState &StateD3D12ProgramBase() const override {
    return state_D3D12ProgramBase_;
  }

 public:
  const ComputeProgramDesc &GetDesc() const override {
    return IBackendComputeProgram::CommonGetDesc();
  }
};
}  // namespace grassland::rhi::backend::d3d12
