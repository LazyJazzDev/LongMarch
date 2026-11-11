#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12Queue;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Queue)
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"

namespace grassland::rhi::backend::d3d12 {

class ID3D12Queue : public Queue {
 public:
  static ID3D12Queue *Create(ID3D12Device *device, QueueType type);
  virtual rhi::ICommandContext *CreateCommandContext() = 0;
  virtual ID3D12CommandQueue *NativeQueue() const = 0;
  virtual ID3D12Timeline *GetTimeline() const = 0;
  virtual bool WaitIdle() = 0;
};

;

}  // namespace grassland::rhi::backend::d3d12
