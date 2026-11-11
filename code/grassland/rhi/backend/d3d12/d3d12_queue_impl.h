#pragma once
#include "grassland/rhi/backend/backend_queue_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12QueueImpl final : public Object<ID3D12Queue> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IQueue, ID3D12Queue, Queue, rhi::IQueue)
  rhi::ICommandContext *CreateCommandContext() override;
  static ID3D12Queue *Create(ID3D12Device *device, QueueType type);
  ID3D12CommandQueue *NativeQueue() const;
  ID3D12Timeline *GetTimeline() const;
  bool WaitIdle();

 private:
  D3D12QueueImpl(ID3D12Device *device, QueueType type);
  ComPtr<ID3D12CommandQueue> queue_;
  ObjectPtr<ID3D12Timeline> timeline_;

 public:
  ~D3D12QueueImpl() override {
  }

 private:
  QueueState state_Queue_;

  QueueState &StateQueue() override {
    return state_Queue_;
  }

  const QueueState &StateQueue() const override {
    return state_Queue_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
