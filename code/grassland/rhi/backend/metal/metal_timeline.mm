#import <Metal/Metal.h>
#include "grassland/rhi/backend/metal/metal_timeline_impl.h"

#include <condition_variable>
#include <mutex>

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"

namespace grassland::rhi::backend::metal {
struct IMetalTimeline::State {
  std::mutex mutex;
  std::condition_variable changed;
  bool failed{};
};

MetalTimelineImpl::MetalTimelineImpl(IMetalDevice *device) : state_(std::make_shared<State>()) {
  InitializeTimeline(device);
}

ObjectPtr<IMetalTimeline> MetalTimelineImpl::Create(IMetalDevice *device) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result = ObjectPtr<IMetalTimeline>::Adopt(new MetalTimelineImpl(device));
  result->gpu_event_ = NS::TransferPtr(device->NativeDevice()->newEvent());
  result->event_ = NS::TransferPtr(device->NativeDevice()->newSharedEvent());
  if (!result->gpu_event_ || !result->event_) {
    ReportError(-1, "Create Metal timeline failed.");
    return {};
  }
  return result;
}

bool MetalTimelineImpl::GetCompletedValue(uint64_t &value) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  value = event_->signaledValue();
  return !state_->failed;
}

bool MetalTimelineImpl::WaitHost(uint64_t value) const {
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

MTL::Event *MetalTimelineImpl::NativeEvent() const {
  return gpu_event_.get();
}

MTL::SharedEvent *MetalTimelineImpl::HostEvent() const {
  return event_.get();
}

void MetalTimelineImpl::ObserveCompletion(MTL::CommandBuffer *commands, uint64_t value, std::function<void()> notify) {
  auto state = state_;
  auto host_event = event_;
  [(__bridge id<MTLCommandBuffer>)(void *)commands addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      if (completed.status == MTLCommandBufferStatusError) {
        state->failed = true;
        state->changed.notify_all();
      } else if (host_event->signaledValue() < value) {
        // GPU dependencies use a separate MTLEvent. Publish host progress only
        // after successful execution, without a GPU-side SharedEvent signal.
        host_event->setSignaledValue(value);
      }
    }
    notify();
  }];
}

void MetalTimelineImpl::MarkFailed() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->failed = true;
  state_->changed.notify_all();
}

}  // namespace grassland::rhi::backend::metal
