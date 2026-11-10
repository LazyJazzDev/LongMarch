#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"

namespace grassland::rhi::backend::metal {

class MetalQueue final : public Queue {
 public:
  rhi::ICommandContext *CreateCommandContext() override;
  static MetalQueue *Create(MetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor);
  MTL::CommandQueue *NativeQueue() const;
  NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer();
  bool WaitIdle();
  MetalTimeline *GetTimeline() const override;

 private:
  MetalQueue(MetalDevice *device, QueueType type);
  NS::SharedPtr<MTL::CommandQueue> queue_;
  std::unique_ptr<MetalTimeline> timeline_;
  NS::SharedPtr<MTL::CommandBufferDescriptor> descriptor_;
};

}  // namespace grassland::rhi::backend::metal
