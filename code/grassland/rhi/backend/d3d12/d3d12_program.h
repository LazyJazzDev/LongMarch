#pragma once
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12Program final : public Program {
 public:
  D3D12Program(D3D12Device *device, const ProgramDesc &desc);
  bool Finalize() override;
  ID3D12PipelineState *NativePipeline() const;
  ID3D12RootSignature *NativeRootSignature() const;

 private:
  ComPtr<ID3D12PipelineState> pipeline_;
  ComPtr<ID3D12RootSignature> root_signature_;
};
}  // namespace grassland::rhi::backend::d3d12
