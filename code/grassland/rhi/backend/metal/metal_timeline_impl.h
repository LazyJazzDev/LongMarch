#pragma once
#include "grassland/rhi/backend/backend_timeline_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"

namespace grassland::rhi::backend::metal {
class MetalTimelineImpl final : public Object<IMetalTimeline> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITimeline, IMetalTimeline, IBackendTimeline, ITimeline)
  static ObjectPtr<IMetalTimeline> Create(IMetalDevice *device);
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  MTL::Event *NativeEvent() const;
  MTL::SharedEvent *HostEvent() const;
  void MarkFailed();
  void ObserveCompletion(MTL::CommandBuffer *commands, uint64_t value, std::function<void()> notify);

 private:
  explicit MetalTimelineImpl(IMetalDevice *device);
  struct State;
  std::shared_ptr<State> state_;
  NS::SharedPtr<MTL::Event> gpu_event_;
  NS::SharedPtr<MTL::SharedEvent> event_;

 public:
  ~MetalTimelineImpl() override {
  }

 private:
  TimelineState state_Timeline_;

  TimelineState &StateTimeline() override {
    return state_Timeline_;
  }

  const TimelineState &StateTimeline() const override {
    return state_Timeline_;
  }
};
}  // namespace grassland::rhi::backend::metal
