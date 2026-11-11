#import <Metal/Metal.h>

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_queue.h"

namespace grassland::rhi::backend::metal {

void MetalDevice::InterruptCollector() {
  std::lock_guard<std::mutex> lock(wait_state_->mutex);
  wait_state_->notified = true;
  wait_state_->changed.notify_all();
}

void MetalDevice::NotifyTimelineFailure() {
  std::lock_guard<std::mutex> lock(wait_state_->mutex);
  wait_state_->changed.notify_all();
}

void MetalDevice::ObserveCompletion(MTL::CommandBuffer *commands, Event signal) {
  auto state = wait_state_;
  rhi::QueryInterfacePtr<MetalTimeline>(signal.timeline).Get()->ObserveCompletion(commands, signal.value, [state] {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->notified = true;
    state->changed.notify_all();
  });
}

bool MetalDevice::WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto state = wait_state_;
  for (const auto &event : events) {
    auto native =
        (__bridge id<MTLSharedEvent>)(void *)rhi::QueryInterfacePtr<MetalTimeline>(event.timeline).Get()->HostEvent();
    std::weak_ptr<WaitState> weak = state;
    [native notifyListener:(__bridge MTLSharedEventListener *)(void *)state->listener.get()
                   atValue:event.value
                     block:^(id<MTLSharedEvent>, uint64_t) {
                       if (auto alive = weak.lock()) {
                         std::lock_guard<std::mutex> lock(alive->mutex);
                         alive->notified = true;
                         alive->changed.notify_all();
                       }
                     }];
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  for (;;) {
    size_t ready = 0, failed = 0;
    for (const auto &event : events) {
      uint64_t value;
      if (!event.timeline->GetCompletedValue(value))
        ++failed;
      else if (value >= event.value)
        ++ready;
    }
    if ((wait_all && ready == events.size()) || (!wait_all && ready))
      return true;
    if ((wait_all && failed) || failed == events.size())
      return false;
    state->changed.wait(lock);
  }
}

void MetalDevice::WaitForProgress(const std::vector<Event> &) {
  auto state = wait_state_;
  std::unique_lock<std::mutex> lock(state->mutex);
  state->changed.wait(lock, [&] { return state->notified; });
  state->notified = false;
}

}  // namespace grassland::rhi::backend::metal
