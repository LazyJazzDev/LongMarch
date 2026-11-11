#include "grassland/rhi/backend/metal/metal_queue.h"

#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_command_context_impl.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_queue_impl.h"
#include "grassland/rhi/backend/metal/metal_timeline_impl.h"

namespace grassland::rhi::backend::metal {

rhi::ICommandContext *MetalQueueImpl::CreateCommandContext() {
  return IMetalCommandContext::Create(this);
}

IMetalQueue *MetalQueueImpl::Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor) {
  auto result = std::unique_ptr<MetalQueueImpl>(new MetalQueueImpl(device, type));
  result->queue_ = NS::TransferPtr(device->NativeDevice()->newCommandQueue());
  if (!result->queue_) {
    ReportError(-1, "Metal command queue creation failed.");
    return nullptr;
  }
  result->descriptor_ = NS::RetainPtr(descriptor);
  result->timeline_ = IMetalTimeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

IMetalTimeline *MetalQueueImpl::GetTimeline() const {
  return timeline_.Get();
}

MTL::CommandQueue *MetalQueueImpl::NativeQueue() const {
  return queue_.get();
}

NS::SharedPtr<MTL::CommandBuffer> MetalQueueImpl::CreateCommandBuffer() {
  return NS::RetainPtr(queue_->commandBuffer(descriptor_.get()));
}

bool MetalQueueImpl::WaitIdle() {
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

MetalQueueImpl::MetalQueueImpl(IMetalDevice *device, QueueType type) {
  InitializeQueue(device, type);
}

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalQueue, "b34deab1-e42d-43f3-b6c4-f47dc7638e30")

namespace grassland::rhi::backend::metal {
IMetalQueue *IMetalQueue::Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor) {
  return MetalQueueImpl::Create(device, type, descriptor);
}
}  // namespace grassland::rhi::backend::metal
