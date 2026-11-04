#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
void MetalDevice::TrackCompletion(MTL::CommandBuffer *buffer, QueueType type) {
  buffer->addCompletedHandler(^(MTL::CommandBuffer *completed) {
    std::lock_guard<std::mutex> lock(progress_mutex_);
    failed_[type] = failed_[type] || completed->status() == MTL::CommandBufferStatusError;
    progress_changed_.notify_all();
  });
}
}  // namespace grassland::rhi::backend::metal
