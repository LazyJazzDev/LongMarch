#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalQueue final : public Queue {
 public:
  ICommandContext *CreateCommandContext() override;
  static MetalQueue *Create(MetalDevice *device, QueueType type);
  MTL::CommandQueue *Native() const;
  MTL::SharedEvent *Timeline() const;
  NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer();
  const void *NativeIdentity() const override;

 private:
  MetalQueue(MetalDevice *device, QueueType type, ObjectPtr<MetalTimeline> timeline);
  NS::SharedPtr<MTL::CommandQueue> queue_;
};
}  // namespace grassland::rhi::backend::metal
