#include "grassland/rhi/backend/metal/metal_queue.h"

#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalQueue::MetalQueue(MetalDevice *device, QueueType type) : Queue(device, type) {
}

MetalQueue *MetalQueue::Create(MetalDevice *device, QueueType type) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result = std::unique_ptr<MetalQueue>(new MetalQueue(device, type));
  result->queue_ = NS::TransferPtr(device->NativeDevice()->newCommandQueue());
  result->timeline_ = NS::TransferPtr(device->NativeDevice()->newSharedEvent());
  if (!result->queue_ || !result->timeline_) {
    ReportError(-1, "Create Metal queue timeline failed.");
    return nullptr;
  }
  return result.release();
}

MTL::CommandQueue *MetalQueue::Native() const {
  return queue_.get();
}

MTL::SharedEvent *MetalQueue::Timeline() const {
  return timeline_.get();
}

const void *MetalQueue::NativeIdentity() const {
  return queue_.get();
}

bool MetalQueue::QueryTimeline(uint64_t &value) const {
  value = timeline_->signaledValue();
  return !failed_.load();
}

void MetalQueue::RecordFailure() {
  failed_.store(true);
}

NS::SharedPtr<MTL::CommandBuffer> MetalQueue::CreateCommandBuffer() {
  return NS::RetainPtr(queue_->commandBuffer(static_cast<MetalDevice *>(Owner())->CommandBufferDescriptor()));
}

ICommandContext *MetalQueue::CreateCommandContext() {
  return MetalCommandContext::Create(this);
}

}  // namespace grassland::rhi::backend::metal
