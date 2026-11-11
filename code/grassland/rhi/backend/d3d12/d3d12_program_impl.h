#pragma once
#include "grassland/rhi/backend/backend_program_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base_state.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12ProgramImpl final : public Object<ID3D12Program> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IProgram, ID3D12Program, Program, IProgram, D3D12ProgramBase, IObject)
  static ID3D12Program *Create(ID3D12Device *device, const ProgramDesc &desc);
  ID3D12PipelineState *NativePipeline() const;

 private:
  D3D12ProgramImpl(ID3D12Device *device, const ProgramDesc &desc);
  ComPtr<ID3D12PipelineState> pipeline_;

 public:
  ~D3D12ProgramImpl() override {
  }

 private:
  ProgramState state_Program_;

  ProgramState &StateProgram() override {
    return state_Program_;
  }

  const ProgramState &StateProgram() const override {
    return state_Program_;
  }

 private:
  D3D12ProgramBaseState state_D3D12ProgramBase_;

  D3D12ProgramBaseState &StateD3D12ProgramBase() override {
    return state_D3D12ProgramBase_;
  }

  const D3D12ProgramBaseState &StateD3D12ProgramBase() const override {
    return state_D3D12ProgramBase_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
