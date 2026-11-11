#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12TimelineImpl final : public Object<rhi::ITimeline, ID3D12Timeline> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITimeline, ID3D12Timeline, IBackendTimeline, ITimeline)

  static ObjectPtr<ID3D12Timeline> Create(ID3D12Device *device);

  bool GetCompletedValue(uint64_t &value) const override;

  bool WaitHost(uint64_t value) const override;

  ID3D12Fence *NativeFence() const;

  ~D3D12TimelineImpl() override = default;

  IBackendDevice *Owner() const override {
    return owner_;
  }

 private:
  explicit D3D12TimelineImpl(ID3D12Device *device);

  ComPtr<ID3D12Fence> fence_;

  void InitializeTimeline(IBackendDevice *device) {
    owner_ = device;
  }

  IBackendDevice *owner_{};
};
}  // namespace grassland::rhi::backend::d3d12
