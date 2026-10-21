#pragma once
#include <array>

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
  uint32_t RootParameter(uint32_t group, bool sampler) const;

 private:
  std::vector<std::array<uint32_t, 2>> root_parameters_;
  ComPtr<ID3D12PipelineState> pipeline_;
  ComPtr<ID3D12RootSignature> root_signature_;
};
}  // namespace grassland::rhi::backend::d3d12
