#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12ProgramBase;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::D3D12ProgramBase)
#include <array>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
// Layout ownership only; object identity and reference counting belong to the concrete Program.
struct D3D12ProgramBaseState;

class D3D12ProgramBase : public IObject {
 public:
  ID3D12RootSignature *NativeRootSignature() const;
  uint32_t RootParameter(uint32_t group, bool sampler) const;

 protected:
  void InitializeD3D12ProgramBase();
  void DestroyD3D12ProgramBase();
  bool CreateLayout(::ID3D12Device *device, const PipelineLayoutDesc &layout, D3D12_ROOT_SIGNATURE_FLAGS flags);
  virtual D3D12ProgramBaseState &StateD3D12ProgramBase() = 0;
  virtual const D3D12ProgramBaseState &StateD3D12ProgramBase() const = 0;
};
}  // namespace grassland::rhi::backend::d3d12
