#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalQueue, "b34deab1-e42d-43f3-b6c4-f47dc7638e30")

namespace grassland::rhi::backend::metal {

class IMetalQueue : public IBackendQueue {
 public:
  virtual IMetalDevice *Owner() const override = 0;
  static IMetalQueue *Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor);
  virtual MTL::CommandQueue *NativeQueue() const = 0;
  virtual NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer() = 0;
  virtual bool WaitIdle() = 0;
  virtual IMetalTimeline *GetTimeline() const = 0;
};

}  // namespace grassland::rhi::backend::metal
