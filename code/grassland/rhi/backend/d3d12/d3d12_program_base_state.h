#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"

namespace grassland::rhi::backend::d3d12 {
struct D3D12ProgramBaseState final {
  D3D12ProgramBaseState();
  ~D3D12ProgramBaseState();
  std::vector<std::array<uint32_t, 2>> root_parameters_{};
  ComPtr<ID3D12RootSignature> root_signature_{};
};
}  // namespace grassland::rhi::backend::d3d12
