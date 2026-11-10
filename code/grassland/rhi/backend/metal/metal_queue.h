#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {

class MetalQueue final : public Queue {
 public:
  MetalQueue(MetalDevice *device, QueueType type);
};

}  // namespace grassland::rhi::backend::metal
