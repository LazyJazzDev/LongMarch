#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12Timeline final : public Timeline {
 public:
  static std::unique_ptr<D3D12Timeline> Create(D3D12Device *device);
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  ID3D12Fence *NativeFence() const;

 private:
  explicit D3D12Timeline(D3D12Device *device);
  ComPtr<ID3D12Fence> fence_;
};

}  // namespace grassland::rhi::backend::d3d12
