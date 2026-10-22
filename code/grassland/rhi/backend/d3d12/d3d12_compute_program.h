#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12ComputeProgram final : public ComputeProgram, public D3D12ProgramBase {
 public:
  static D3D12ComputeProgram *Create(D3D12Device *device, const ComputeProgramDesc &desc);
  ID3D12PipelineState *NativePipeline() const;

 private:
  D3D12ComputeProgram(D3D12Device *device, const ComputeProgramDesc &desc);
  ComPtr<ID3D12PipelineState> pipeline_;
};
}  // namespace grassland::rhi::backend::d3d12
