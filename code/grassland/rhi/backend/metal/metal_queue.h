#pragma once
#include <list>
#include <mutex>

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_queue.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalQueue,
                        grassland::rhi::IObject,
                        "b34deab1-e42d-43f3-b6c4-f47dc7638e30")

namespace grassland::rhi::backend::metal {

class IMetalQueue : public IObject {
 public:
  virtual std::mutex &QueueMutex() = 0;
  virtual  // Called under QueueMutex, in native submission order.
      uint64_t
      NextSerial() = 0;
  virtual ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IMetalCommandContext> &commands,
                                                    const SubmitInfo &info) = 0;
  virtual  // Called only by the device collector; returns whether this queue is drained.
      bool
      CollectCompleted(uint64_t completed, bool timeline_valid) = 0;
  virtual IMetalDevice *Owner() const = 0;
  static IMetalQueue *Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor);
  virtual MTL::CommandQueue *NativeQueue() const = 0;
  virtual NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer() = 0;
  virtual bool WaitIdle() = 0;
  virtual IMetalTimeline *GetTimeline() const = 0;
};

}  // namespace grassland::rhi::backend::metal
