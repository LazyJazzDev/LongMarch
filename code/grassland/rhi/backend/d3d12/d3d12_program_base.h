#pragma once
#include <array>

#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12ProgramBase)

namespace grassland::rhi::backend::d3d12 {
// Layout ownership only; object identity and reference counting belong to the concrete Program.
class ID3D12ProgramBase : public IObject {
 public:
  virtual ID3D12RootSignature *NativeRootSignature() const = 0;
  virtual uint32_t RootParameter(uint32_t group, bool sampler) const = 0;
};
}  // namespace grassland::rhi::backend::d3d12
