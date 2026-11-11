#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Queue final : public Queue {
 public:
  LONGMARCH_RHI_QUERY(D3D12Queue, Queue)

  rhi::ICommandContext *CreateCommandContext() override;
  static D3D12Queue *Create(D3D12Device *device, QueueType type);
  ID3D12CommandQueue *NativeQueue() const;
  D3D12Timeline *GetTimeline() const;
  bool WaitIdle();

 private:
  D3D12Queue(D3D12Device *device, QueueType type);
  ComPtr<ID3D12CommandQueue> queue_;
  ObjectPtr<D3D12Timeline> timeline_;
};

}  // namespace grassland::rhi::backend::d3d12
