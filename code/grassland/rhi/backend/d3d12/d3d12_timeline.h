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

namespace grassland::rhi::backend::d3d12 {
class D3D12TimelineImpl final : public Object<rhi::ITimeline, ID3D12Timeline> {
 public:
  static ObjectPtr<ID3D12Timeline> Create(ID3D12Device *device);

  bool GetCompletedValue(uint64_t &value) const override;

  bool WaitHost(uint64_t value) const override;

  ID3D12Fence *NativeFence() const;

  ~D3D12TimelineImpl() override = default;

  ID3D12Device *Owner() const override {
    return owner_;
  }

 private:
  explicit D3D12TimelineImpl(ID3D12Device *device);

  ComPtr<ID3D12Fence> fence_;

  void InitializeTimeline(ID3D12Device *device) {
    owner_ = device;
  }

  ID3D12Device *owner_{};
};
}  // namespace grassland::rhi::backend::d3d12
