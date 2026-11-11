#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class IMetalQueue;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalQueue)
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"

namespace grassland::rhi::backend::metal {

class IMetalQueue : public Queue {
 public:
  static IMetalQueue *Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor);
  virtual rhi::ICommandContext *CreateCommandContext() = 0;
  virtual MTL::CommandQueue *NativeQueue() const = 0;
  virtual NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer() = 0;
  virtual bool WaitIdle() = 0;
  virtual IMetalTimeline *GetTimeline() const = 0;
};

;

}  // namespace grassland::rhi::backend::metal
