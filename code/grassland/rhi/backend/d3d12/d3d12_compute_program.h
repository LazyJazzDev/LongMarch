#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"
#include "grassland/rhi/rhi_compute_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12ComputeProgram,
                        grassland::rhi::backend::d3d12::ID3D12ProgramBase,
                        "f0ee363d-3b36-4a53-8e17-b8a0ea11bf0a")

namespace grassland::rhi::backend::d3d12 {
class ID3D12ComputeProgram : public ID3D12ProgramBase {
 public:
  static rhi::IComputeProgram *Create(ID3D12Device *device, const ComputeProgramDesc &desc);
  virtual ID3D12PipelineState *NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12
