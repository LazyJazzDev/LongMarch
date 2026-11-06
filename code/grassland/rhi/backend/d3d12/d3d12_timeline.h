#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12Timeline final : public backend::Timeline {
 public:
  static D3D12Timeline *Create(D3D12Device *device);
  ~D3D12Timeline() override;
  ID3D12Fence *Native() const;
  bool GetCompletedValue(uint64_t &value) const override;

  bool IsNative() const override {
    return true;
  }

  bool Wait(uint64_t value, HANDLE event) const;

 private:
  explicit D3D12Timeline(D3D12Device *device);
  ComPtr<ID3D12Fence> fence_;
};
}  // namespace grassland::rhi::backend::d3d12
