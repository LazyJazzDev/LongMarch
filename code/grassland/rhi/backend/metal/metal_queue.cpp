#include "grassland/rhi/backend/metal/metal_queue.h"

#include "grassland/rhi/backend/metal/metal_command_context.h"

namespace grassland::rhi::backend::metal {

rhi::ICommandContext *MetalQueue::CreateCommandContext() {
  return MetalCommandContext::Create(this);
}

MetalQueue::MetalQueue(MetalDevice *device, QueueType type) : Queue(device, type) {
}

}  // namespace grassland::rhi::backend::metal
