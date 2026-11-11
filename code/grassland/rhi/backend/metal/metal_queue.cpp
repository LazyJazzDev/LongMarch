#include "grassland/rhi/backend/metal/metal_queue.h"

#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"

namespace grassland::rhi::backend::metal {

rhi::ICommandContext *MetalQueue::CreateCommandContext() {
  return MetalCommandContext::Create(this);
}

MetalQueue *MetalQueue::Create(MetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor) {
  auto result = std::unique_ptr<MetalQueue>(new MetalQueue(device, type));
  result->queue_ = NS::TransferPtr(device->NativeDevice()->newCommandQueue());
  if (!result->queue_) {
    ReportError(-1, "Metal command queue creation failed.");
    return nullptr;
  }
  result->descriptor_ = NS::RetainPtr(descriptor);
  result->timeline_ = MetalTimeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

MetalTimeline *MetalQueue::GetTimeline() const {
  return timeline_.Get();
}

MTL::CommandQueue *MetalQueue::NativeQueue() const {
  return queue_.get();
}

NS::SharedPtr<MTL::CommandBuffer> MetalQueue::CreateCommandBuffer() {
  return NS::RetainPtr(queue_->commandBuffer(descriptor_.get()));
}

bool MetalQueue::WaitIdle() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto marker = CreateCommandBuffer();
  if (!marker) {
    ReportError(-1, "Metal idle marker allocation failed.");
    return false;
  }
  marker->commit();
  marker->waitUntilCompleted();
  if (marker->status() != MTL::CommandBufferStatusCompleted) {
    ReportMetalError(marker->error(), "Metal idle marker failed",
                     descriptor_->errorOptions() != MTL::CommandBufferErrorOptionNone);
    return false;
  }
  return true;
}

MetalQueue::MetalQueue(MetalDevice *device, QueueType type) : Queue(device, type) {
}

}  // namespace grassland::rhi::backend::metal
