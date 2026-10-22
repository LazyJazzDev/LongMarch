#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12Program final : public Program, public D3D12ProgramBase {
 public:
  static D3D12Program *Create(D3D12Device *device, const ProgramDesc &desc);
  ID3D12PipelineState *NativePipeline() const;

 private:
  D3D12Program(D3D12Device *device, const ProgramDesc &desc);
  ComPtr<ID3D12PipelineState> pipeline_;
};
}  // namespace grassland::rhi::backend::d3d12
