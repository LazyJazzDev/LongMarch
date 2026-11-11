#pragma once
#include "grassland/rhi/backend/metal/metal_timeline.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalTimelineImpl final : public Object<rhi::ITimeline, IMetalTimeline> {
 public:
  static ObjectPtr<IMetalTimeline> Create(IMetalDevice *device);

  bool GetCompletedValue(uint64_t &value) const override;

  bool WaitHost(uint64_t value) const override;

  MTL::Event *NativeEvent() const;

  MTL::SharedEvent *HostEvent() const;

  void MarkFailed();

  void ObserveCompletion(MTL::CommandBuffer *commands, uint64_t value, std::function<void()> notify);

  ~MetalTimelineImpl() override = default;

  IBackendDevice *Owner() const override {
    return owner_;
  }

 private:
  explicit MetalTimelineImpl(IMetalDevice *device);

  struct CompletionState;

  std::shared_ptr<CompletionState> state_;

  NS::SharedPtr<MTL::Event> gpu_event_;

  NS::SharedPtr<MTL::SharedEvent> event_;

  void InitializeTimeline(IBackendDevice *device) {
    owner_ = device;
  }

  IBackendDevice *owner_{};
};
}  // namespace grassland::rhi::backend::metal
