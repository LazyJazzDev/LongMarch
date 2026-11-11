#pragma once
#include "grassland/rhi/backend/backend_timeline_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12TimelineImpl final : public Object<ID3D12Timeline> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITimeline, ID3D12Timeline, Timeline, ITimeline)
  static ObjectPtr<ID3D12Timeline> Create(ID3D12Device *device);
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  ID3D12Fence *NativeFence() const;

 private:
  explicit D3D12TimelineImpl(ID3D12Device *device);
  ComPtr<ID3D12Fence> fence_;

 public:
  ~D3D12TimelineImpl() override {
  }

 private:
  TimelineState state_Timeline_;

  TimelineState &StateTimeline() override {
    return state_Timeline_;
  }

  const TimelineState &StateTimeline() const override {
    return state_Timeline_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
