#pragma once
#include <array>

#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
// Layout ownership only; object identity and reference counting belong to the concrete Program.
class D3D12ProgramBase {
 public:
  ID3D12RootSignature *NativeRootSignature() const;
  uint32_t RootParameter(uint32_t group, bool sampler) const;

 protected:
  D3D12ProgramBase();
  ~D3D12ProgramBase();
  D3D12ProgramBase(const D3D12ProgramBase &) = delete;
  D3D12ProgramBase &operator=(const D3D12ProgramBase &) = delete;
  bool CreateLayout(ID3D12Device *device, const PipelineLayoutDesc &layout, D3D12_ROOT_SIGNATURE_FLAGS flags);

 private:
  std::vector<std::array<uint32_t, 2>> root_parameters_;
  ComPtr<ID3D12RootSignature> root_signature_;
};
}  // namespace grassland::rhi::backend::d3d12
