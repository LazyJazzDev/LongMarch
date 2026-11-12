#pragma once
#include <list>
#include <mutex>

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"
#include "grassland/rhi/rhi_queue.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12Queue,
                        grassland::rhi::IObject,
                        "46194ddb-7a1a-4d51-b878-0e51d212d642")

namespace grassland::rhi::backend::d3d12 {

class ID3D12Queue : public IObject {
 public:
  virtual std::mutex &QueueMutex() = 0;
  virtual  // Called under QueueMutex, in native submission order.
      uint64_t
      NextSerial() = 0;
  virtual ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<ID3D12CommandContext> &commands,
                                                    const SubmitInfo &info) = 0;
  virtual  // Called only by the device collector; returns whether this queue is drained.
      bool
      CollectCompleted(uint64_t completed, bool timeline_valid) = 0;
  virtual ID3D12Device *Owner() const = 0;
  static ID3D12Queue *Create(ID3D12Device *device, QueueType type);
  virtual ID3D12CommandQueue *NativeQueue() const = 0;
  virtual ID3D12Timeline *GetTimeline() const = 0;
  virtual bool WaitIdle() = 0;
};

}  // namespace grassland::rhi::backend::d3d12
