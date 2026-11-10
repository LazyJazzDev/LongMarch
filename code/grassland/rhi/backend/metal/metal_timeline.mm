#import <Metal/Metal.h>

#include <condition_variable>
#include <mutex>

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"

namespace grassland::rhi::backend::metal {
struct MetalTimeline::State {
  std::mutex mutex;
  std::condition_variable changed;
  bool failed{};
};

MetalTimeline::MetalTimeline(MetalDevice *device) : Timeline(device), state_(std::make_shared<State>()) {
}

std::unique_ptr<MetalTimeline> MetalTimeline::Create(MetalDevice *device) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result = std::unique_ptr<MetalTimeline>(new MetalTimeline(device));
  result->event_ = NS::TransferPtr(device->NativeDevice()->newSharedEvent());
  if (!result->event_) {
    ReportError(-1, "Create Metal timeline failed.");
    return {};
  }
  return result;
}

bool MetalTimeline::GetCompletedValue(uint64_t &value) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  value = event_->signaledValue();
  return !state_->failed;
}

bool MetalTimeline::WaitHost(uint64_t value) const {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto listener = NS::TransferPtr(MTL::SharedEventListener::alloc()->init());
  if (!listener)
    return false;
  auto state = state_;
  auto native = (__bridge id<MTLSharedEvent>)(void *)event_.get();
  [native notifyListener:(__bridge MTLSharedEventListener *)(void *)listener.get()
                 atValue:value
                   block:^(id<MTLSharedEvent>, uint64_t) {
                     std::lock_guard<std::mutex> lock(state->mutex);
                     state->changed.notify_all();
                   }];
  std::unique_lock<std::mutex> lock(state->mutex);
  state->changed.wait(lock, [&] { return state->failed || event_->signaledValue() >= value; });
  return !state->failed;
}

MTL::SharedEvent *MetalTimeline::NativeEvent() const {
  return event_.get();
}

void MetalTimeline::MarkFailed() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->failed = true;
  state_->changed.notify_all();
}

}  // namespace grassland::rhi::backend::metal
