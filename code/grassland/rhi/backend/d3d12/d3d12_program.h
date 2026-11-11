#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12Program, "2919454c-7ff9-4ce3-b919-40901e038398")

namespace grassland::rhi::backend::d3d12 {
class ID3D12Program : public ID3D12ProgramBase {
 public:
  static rhi::IProgram *Create(ID3D12Device *device, const ProgramDesc &desc);
  virtual ID3D12PipelineState *NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12
