#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12Queue final : public Queue {
 public:
  ICommandContext *CreateCommandContext() override;
  static D3D12Queue *Create(D3D12Device *device, QueueType type);
  ID3D12CommandQueue *Native() const;
  ID3D12Fence *Timeline() const;
  bool Wait(uint64_t value, HANDLE event) const;
  bool WaitIdle();
  const void *NativeIdentity() const override;

  bool HasNativeTimeline() const override {
    return true;
  }

  bool QueryTimeline(uint64_t &value) const override;

 private:
  D3D12Queue(D3D12Device *device, QueueType type);
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<ID3D12Fence> timeline_;
};
}  // namespace grassland::rhi::backend::d3d12
