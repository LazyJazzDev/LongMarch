#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Timeline)

namespace grassland::rhi::backend::d3d12 {
class ID3D12Timeline : public IBackendTimeline {
 public:
  static ObjectPtr<ID3D12Timeline> Create(ID3D12Device *device);
  virtual bool GetCompletedValue(uint64_t &value) const = 0;
  virtual bool WaitHost(uint64_t value) const = 0;
  virtual ID3D12Fence *NativeFence() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12
