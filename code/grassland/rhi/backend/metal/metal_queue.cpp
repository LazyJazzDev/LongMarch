#include "grassland/rhi/backend/metal/metal_queue.h"

namespace grassland::rhi::backend::metal {

MetalQueue::MetalQueue(MetalDevice *device, QueueType type) : Queue(device, type) {
}

}  // namespace grassland::rhi::backend::metal
