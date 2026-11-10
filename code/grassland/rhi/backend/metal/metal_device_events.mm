#import <Metal/Metal.h>

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_queue.h"

namespace grassland::rhi::backend::metal {

uint64_t MetalDevice::ProgressGeneration() const {
  std::lock_guard<std::mutex> lock(wait_state_->mutex);
  return wait_state_->generation;
}

void MetalDevice::NotifyCollector() {
  std::lock_guard<std::mutex> lock(wait_state_->mutex);
  ++wait_state_->generation;
  wait_state_->changed.notify_all();
}

void MetalDevice::ObserveCompletion(MTL::CommandBuffer *commands) {
  auto state = wait_state_;
  [(__bridge id<MTLCommandBuffer>)(void *)commands addCompletedHandler:^(id<MTLCommandBuffer>) {
    std::lock_guard<std::mutex> lock(state->mutex);
    ++state->generation;
    state->changed.notify_all();
  }];
}

bool MetalDevice::WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto state = wait_state_;
  for (const auto &event : events) {
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

void MetalDevice::WaitForProgress(const std::vector<Event> &next, uint64_t observed) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto state = wait_state_;
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
  state->changed.wait(lock, [&] { return state->generation != observed; });
}

}  // namespace grassland::rhi::backend::metal
