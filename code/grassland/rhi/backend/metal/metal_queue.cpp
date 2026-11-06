#include "grassland/rhi/backend/metal/metal_queue.h"

#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalQueue::MetalQueue(MetalDevice *device, QueueType type, ObjectPtr<MetalTimeline> timeline)
    : Queue(device, type, std::move(timeline)) {
}

MetalQueue *MetalQueue::Create(MetalDevice *device, QueueType type) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto timeline = ObjectPtr<MetalTimeline>::Adopt(MetalTimeline::Create(device));
  if (!timeline)
    return nullptr;
  auto result = std::unique_ptr<MetalQueue>(new MetalQueue(device, type, std::move(timeline)));
  result->queue_ = NS::TransferPtr(device->NativeDevice()->newCommandQueue());
  if (!result->queue_) {
    ReportError(-1, "Create Metal queue timeline failed.");
    return nullptr;
  }
  return result.release();
}

MTL::CommandQueue *MetalQueue::Native() const {
  return queue_.get();
}

MTL::SharedEvent *MetalQueue::Timeline() const {
  return static_cast<MetalTimeline *>(TimelineState())->Native();
}

const void *MetalQueue::NativeIdentity() const {
  return queue_.get();
}

NS::SharedPtr<MTL::CommandBuffer> MetalQueue::CreateCommandBuffer() {
  return NS::RetainPtr(queue_->commandBuffer(static_cast<MetalDevice *>(Owner())->CommandBufferDescriptor()));
}

ICommandContext *MetalQueue::CreateCommandContext() {
  return MetalCommandContext::Create(this);
}

}  // namespace grassland::rhi::backend::metal
