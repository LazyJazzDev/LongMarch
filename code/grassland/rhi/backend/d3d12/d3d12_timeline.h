#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12Timeline;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Timeline)
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12Device;

class ID3D12Timeline : public Timeline {
 public:
  static ObjectPtr<ID3D12Timeline> Create(ID3D12Device *device);
  virtual bool GetCompletedValue(uint64_t &value) const = 0;
  virtual bool WaitHost(uint64_t value) const = 0;
  virtual ID3D12Fence *NativeFence() const = 0;
};

;

}  // namespace grassland::rhi::backend::d3d12
