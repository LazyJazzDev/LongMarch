#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_queue.h"

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

  bool HasNativeTimeline() const override {
    return true;
  }

  bool QueryTimeline(uint64_t &value) const override;
  void RecordFailure();

 private:
  MetalQueue(MetalDevice *device, QueueType type);
  NS::SharedPtr<MTL::CommandQueue> queue_;
  NS::SharedPtr<MTL::SharedEvent> timeline_;
  std::atomic<bool> failed_{};
};
}  // namespace grassland::rhi::backend::metal
