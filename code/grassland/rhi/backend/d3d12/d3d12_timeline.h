#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_util.h"
#include "grassland/rhi/rhi_timeline.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12Timeline,
                        grassland::rhi::IObject,
                        "32d5dccb-60c3-4ca2-af93-4b67d85e7b16")

namespace grassland::rhi::backend::d3d12 {
class ID3D12Timeline : public IObject {
 public:
  virtual ID3D12Device *Owner() const = 0;
  static ObjectPtr<ID3D12Timeline> Create(ID3D12Device *device);
  virtual ID3D12Fence *NativeFence() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12
