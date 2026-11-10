#import <Metal/Metal.h>

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_queue.h"

namespace grassland::rhi::backend::metal {

bool MetalDevice::WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const {
  auto state = wait_state_;
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
    // Queue progress listeners are shared with the collector. A bounded check
    // also observes command-buffer failures, which do not signal SharedEvent.
    state->changed.wait_for(lock, std::chrono::milliseconds(50));
  }
}

void MetalDevice::WaitForProgress(const std::vector<Event> &next) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto state = wait_state_;
  uint64_t observed;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    observed = state->generation;
  }
  for (const auto &event : next) {
    for (size_t index = 0; index < queues_.size(); ++index) {
      if (!queues_[index] || queues_[index]->GetTimeline() != event.timeline)
        continue;
      if (state->values[index] == event.value)
        break;
      state->values[index] = event.value;
      auto native = (__bridge id<MTLSharedEvent>)(void *)static_cast<MetalTimeline *>(event.timeline)->NativeEvent();
      std::weak_ptr<WaitState> weak = state;
      [native notifyListener:(__bridge MTLSharedEventListener *)(void *)state->listener.get()
                     atValue:event.value
                       block:^(id<MTLSharedEvent>, uint64_t) {
                         if (auto alive = weak.lock()) {
                           std::lock_guard<std::mutex> lock(alive->mutex);
                           ++alive->generation;
                           alive->changed.notify_all();
                         }
                       }];
      break;
    }
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  state->changed.wait_for(lock, std::chrono::milliseconds(50), [&] { return state->generation != observed; });
}

}  // namespace grassland::rhi::backend::metal
