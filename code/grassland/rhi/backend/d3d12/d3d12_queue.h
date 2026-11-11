#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12Queue, "46194ddb-7a1a-4d51-b878-0e51d212d642")

namespace grassland::rhi::backend::d3d12 {

class ID3D12Queue : public IBackendQueue {
 public:
  virtual ID3D12Device *Owner() const override = 0;
  static ID3D12Queue *Create(ID3D12Device *device, QueueType type);
  virtual ID3D12CommandQueue *NativeQueue() const = 0;
  virtual ID3D12Timeline *GetTimeline() const = 0;
  virtual bool WaitIdle() = 0;
};

}  // namespace grassland::rhi::backend::d3d12
