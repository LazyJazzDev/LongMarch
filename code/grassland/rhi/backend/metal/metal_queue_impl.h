#pragma once
#include "grassland/rhi/backend/backend_queue_state.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalQueueImpl final : public Object<IMetalQueue> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IQueue, IMetalQueue, IBackendQueue, rhi::IQueue)
  rhi::ICommandContext *CreateCommandContext() override;
  static IMetalQueue *Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor);
  MTL::CommandQueue *NativeQueue() const;
  NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer();
  bool WaitIdle();
  IMetalTimeline *GetTimeline() const override;

 private:
  MetalQueueImpl(IMetalDevice *device, QueueType type);
  NS::SharedPtr<MTL::CommandQueue> queue_;
  ObjectPtr<IMetalTimeline> timeline_;
  NS::SharedPtr<MTL::CommandBufferDescriptor> descriptor_;

 public:
  ~MetalQueueImpl() override {
  }

 private:
  QueueState state_Queue_;

  QueueState &StateQueue() override {
    return state_Queue_;
  }

  const QueueState &StateQueue() const override {
    return state_Queue_;
  }

 public:
  QueueType GetType() const override {
    return IBackendQueue::CommonGetType();
  }
};
}  // namespace grassland::rhi::backend::metal
