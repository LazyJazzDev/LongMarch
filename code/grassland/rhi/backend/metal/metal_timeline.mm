#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_queue.h"

namespace grassland::rhi::backend::metal {
void MetalDevice::TrackCompletion(MTL::CommandBuffer *buffer, Queue *owner) {
  auto queue = static_cast<MetalQueue *>(owner);
  buffer->addCompletedHandler(^(MTL::CommandBuffer *completed) {
    std::lock_guard<std::mutex> lock(progress_mutex_);
    if (completed->status() == MTL::CommandBufferStatusError)
      queue->RecordFailure();
    progress_changed_.notify_all();
  });
}
}  // namespace grassland::rhi::backend::metal
